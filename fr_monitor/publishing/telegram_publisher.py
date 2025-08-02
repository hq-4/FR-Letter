"""
Telegram bot integration for publishing Federal Register summaries.
"""

import requests
from typing import List, Optional, Dict, Any
import structlog
from datetime import datetime
import html

from ..core.models import FinalSummary, PublishingResult
from ..core.config import settings

logger = structlog.get_logger(__name__)


class TelegramPublisher:
    """Publisher for Telegram channel via bot API."""
    
    def __init__(self, bot_token: Optional[str] = None, channel_id: Optional[str] = None):
        self.bot_token = bot_token or settings.telegram_bot_token
        self.channel_id = channel_id or settings.telegram_channel_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json"
        })
        
        # Message length limits
        self.max_message_length = 4096
        self.max_caption_length = 1024
    
    def publish_daily_digest(self, summaries: List[FinalSummary]) -> List[PublishingResult]:
        """
        Publish daily digest to Telegram channel.
        
        Args:
            summaries: List of final summaries to publish
            
        Returns:
            List of PublishingResult objects (one per message sent)
        """
        if not summaries:
            logger.warning("No summaries provided for Telegram publishing")
            return [PublishingResult(
                document_id="daily_digest",
                channel="telegram",
                success=False,
                error_message="No summaries to publish"
            )]
        
        results = []
        
        try:
            # Send header message
            header_result = self._send_header_message()
            results.append(header_result)
            
            # Send each summary as a separate message
            for i, summary in enumerate(summaries, 1):
                result = self._send_summary_message(summary, i)
                results.append(result)
                
                # Small delay between messages to avoid rate limiting
                import time
                time.sleep(0.5)
            
            # Send footer message with highlights
            footer_result = self._send_footer_message(summaries)
            results.append(footer_result)
            
            successful_sends = sum(1 for r in results if r.success)
            logger.info("Completed Telegram publishing", 
                       successful=successful_sends,
                       total=len(results))
            
        except Exception as e:
            logger.error("Failed to publish to Telegram", error=str(e))
            results.append(PublishingResult(
                document_id="daily_digest",
                channel="telegram",
                success=False,
                error_message=str(e)
            ))
        
        return results
    
    def _send_header_message(self) -> PublishingResult:
        """Send header message for daily digest."""
        today = datetime.now().strftime("%B %d, %Y")
        
        message = f"""🏛️ **Federal Register Daily Brief**
📅 {today}

Today's top regulatory developments, analyzed for policy professionals:
        
👇 Thread below with key highlights"""
        
        return self._send_message(message, "header")
    
    def _send_summary_message(self, summary: FinalSummary, index: int) -> PublishingResult:
        """Send individual summary message."""
        # Format message with Telegram markdown
        message_parts = [
            f"**{index}. {summary.headline}**",
            ""
        ]
        
        for bullet in summary.bullets:
            message_parts.append(f"• {bullet}")
        
        message = "\n".join(message_parts)
        
        # Ensure message fits within Telegram limits
        if len(message) > self.max_message_length:
            message = message[:self.max_message_length - 3] + "..."
        
        return self._send_message(message, summary.document_id)
    
    def _send_footer_message(self, summaries: List[FinalSummary]) -> PublishingResult:
        """Send footer message with tweet-length highlights."""
        # Create tweet-length highlights
        highlights = []
        for summary in summaries[:3]:  # Top 3 summaries
            # Create short version for Twitter-style sharing
            short_headline = summary.headline
            if len(short_headline) > 100:
                short_headline = short_headline[:97] + "..."
            
            highlights.append(f"📌 {short_headline}")
        
        message_parts = [
            "**📱 Tweet-length highlights:**",
            ""
        ]
        message_parts.extend(highlights)
        message_parts.extend([
            "",
            "🔗 Full analysis available in our newsletter",
            "💬 Questions? Reply here or DM us",
            "",
            "#FederalRegister #Policy #Regulation"
        ])
        
        message = "\n".join(message_parts)
        
        return self._send_message(message, "footer")
    
    def _send_message(self, text: str, document_id: str) -> PublishingResult:
        """Send a single message to Telegram channel."""
        try:
            payload = {
                "chat_id": self.channel_id,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }
            
            response = self.session.post(
                f"{self.base_url}/sendMessage",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    message_id = data["result"]["message_id"]
                    logger.debug("Sent Telegram message", 
                               message_id=message_id,
                               document_id=document_id)
                    
                    return PublishingResult(
                        document_id=document_id,
                        channel="telegram",
                        success=True,
                        published_at=datetime.utcnow(),
                        external_id=str(message_id)
                    )
                else:
                    error_msg = data.get("description", "Unknown error")
                    logger.error("Telegram API error", error=error_msg)
                    return PublishingResult(
                        document_id=document_id,
                        channel="telegram",
                        success=False,
                        error_message=error_msg
                    )
            else:
                logger.error("Telegram HTTP error", 
                           status_code=response.status_code,
                           response=response.text[:200])
                return PublishingResult(
                    document_id=document_id,
                    channel="telegram",
                    success=False,
                    error_message=f"HTTP {response.status_code}"
                )
                
        except requests.RequestException as e:
            logger.error("Failed to send Telegram message", 
                        document_id=document_id,
                        error=str(e))
            return PublishingResult(
                document_id=document_id,
                channel="telegram",
                success=False,
                error_message=str(e)
            )
    
    def send_test_message(self) -> bool:
        """Send a test message to verify bot configuration."""
        test_message = "🤖 Federal Register Monitor - Test Message\n\nBot is configured and ready!"
        
        result = self._send_message(test_message, "test")
        
        if result.success:
            logger.info("Telegram test message sent successfully")
            return True
        else:
            logger.error("Telegram test message failed", error=result.error_message)
            return False
    
    def health_check(self) -> bool:
        """Check Telegram bot connectivity and permissions."""
        try:
            # Get bot info
            response = self.session.get(
                f"{self.base_url}/getMe",
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error("Failed to connect to Telegram API", 
                           status_code=response.status_code)
                return False
            
            data = response.json()
            if not data.get("ok"):
                logger.error("Telegram bot authentication failed", 
                           error=data.get("description"))
                return False
            
            bot_info = data["result"]
            logger.info("Telegram health check passed", 
                       bot_username=bot_info.get("username"),
                       bot_name=bot_info.get("first_name"))
            
            # Test channel access
            try:
                chat_response = self.session.get(
                    f"{self.base_url}/getChat",
                    params={"chat_id": self.channel_id},
                    timeout=10
                )
                
                if chat_response.status_code == 200:
                    chat_data = chat_response.json()
                    if chat_data.get("ok"):
                        logger.info("Telegram channel access confirmed", 
                                   channel_id=self.channel_id)
                        return True
                    else:
                        logger.warning("Cannot access Telegram channel", 
                                     error=chat_data.get("description"))
                        return False
                else:
                    logger.warning("Cannot verify Telegram channel access")
                    return False
                    
            except Exception as e:
                logger.warning("Could not verify channel access", error=str(e))
                # Bot info worked, so basic connectivity is OK
                return True
                
        except Exception as e:
            logger.error("Telegram health check failed", error=str(e))
            return False
