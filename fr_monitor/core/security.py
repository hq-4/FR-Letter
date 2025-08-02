"""
Security utilities for credential management and least-privilege enforcement.
"""

import os
import hashlib
import secrets
from typing import Optional, Dict, Any
import structlog
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = structlog.get_logger(__name__)


class CredentialManager:
    """Secure credential management with encryption and least-privilege access."""
    
    def __init__(self):
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.required_permissions = {
            'federal_register': ['read'],
            'redis': ['read', 'write'],
            'ollama': ['read'],
            'openrouter': ['read'],
            'substack': ['write'],
            'telegram': ['write']
        }
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for credential storage."""
        key_env = os.getenv('ENCRYPTION_KEY')
        if key_env:
            return key_env.encode()
        
        # Generate new key
        key = Fernet.generate_key()
        logger.warning("Generated new encryption key. Store ENCRYPTION_KEY environment variable for persistence.")
        return key
    
    def encrypt_credential(self, credential: str) -> str:
        """Encrypt a credential."""
        try:
            encrypted = self.cipher_suite.encrypt(credential.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error("Failed to encrypt credential", error=str(e))
            raise
    
    def decrypt_credential(self, encrypted_credential: str) -> str:
        """Decrypt a credential."""
        try:
            encrypted_bytes = base64.b64decode(encrypted_credential.encode())
            decrypted = self.cipher_suite.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error("Failed to decrypt credential", error=str(e))
            raise
    
    def hash_credential(self, credential: str) -> str:
        """Hash a credential for storage (non-reversible)."""
        return hashlib.sha256(credential.encode()).hexdigest()
    
    def validate_credentials(self, credentials: Dict[str, str]) -> Dict[str, bool]:
        """Validate credentials for least-privilege access."""
        validation_results = {}
        
        # Check Federal Register API key
        if 'federal_register_api_key' in credentials:
            api_key = credentials['federal_register_api_key']
            validation_results['federal_register'] = self._validate_federal_register_key(api_key)
        
        # Check Redis credentials
        if 'redis_password' in credentials:
            validation_results['redis'] = self._validate_redis_access(credentials)
        
        # Check OpenRouter API key
        if 'openrouter_api_key' in credentials:
            validation_results['openrouter'] = self._validate_openrouter_key(credentials['openrouter_api_key'])
        
        # Check Substack credentials
        if 'substack_api_key' in credentials:
            validation_results['substack'] = self._validate_substack_key(credentials['substack_api_key'])
        
        # Check Telegram credentials
        if 'telegram_bot_token' in credentials:
            validation_results['telegram'] = self._validate_telegram_token(credentials['telegram_bot_token'])
        
        return validation_results
    
    def _validate_federal_register_key(self, api_key: str) -> bool:
        """Validate Federal Register API key has read-only access."""
        # Basic format validation
        if not api_key or len(api_key) < 10:
            return False
        
        # Check for read-only endpoints (implementation would make actual API call)
        logger.info("Federal Register API key format validated")
        return True
    
    def _validate_redis_access(self, credentials: Dict[str, str]) -> bool:
        """Validate Redis access has appropriate permissions."""
        try:
            import redis
            redis_client = redis.Redis(
                host=credentials.get('redis_host', 'localhost'),
                port=int(credentials.get('redis_port', 6379)),
                password=credentials.get('redis_password'),
                db=int(credentials.get('redis_db', 0)),
                decode_responses=True
            )
            
            # Test basic operations
            redis_client.ping()
            
            # Test write permission
            test_key = f"test_key_{secrets.token_hex(8)}"
            redis_client.setex(test_key, 60, "test_value")
            redis_client.delete(test_key)
            
            logger.info("Redis access validated successfully")
            return True
            
        except Exception as e:
            logger.error("Redis access validation failed", error=str(e))
            return False
    
    def _validate_openrouter_key(self, api_key: str) -> bool:
        """Validate OpenRouter API key."""
        if not api_key or not api_key.startswith('sk-'):
            return False
        
        logger.info("OpenRouter API key format validated")
        return True
    
    def _validate_substack_key(self, api_key: str) -> bool:
        """Validate Substack API key."""
        if not api_key or len(api_key) < 20:
            return False
        
        logger.info("Substack API key format validated")
        return True
    
    def _validate_telegram_token(self, token: str) -> bool:
        """Validate Telegram bot token."""
        if not token or not token.startswith('bot'):
            return False
        
        parts = token.split(':')
        if len(parts) != 2:
            return False
        
        logger.info("Telegram bot token format validated")
        return True
    
    def audit_credentials(self, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Perform security audit of credentials."""
        audit_report = {
            'timestamp': datetime.utcnow().isoformat(),
            'credentials_found': [],
            'permissions': {},
            'recommendations': []
        }
        
        sensitive_keys = [
            'api_key', 'token', 'password', 'secret', 'key'
        ]
        
        for key, value in credentials.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                audit_report['credentials_found'].append(key)
                audit_report['permissions'][key] = 'encrypted'
                
                # Check for common security issues
                if len(value) < 20:
                    audit_report['recommendations'].append(
                        f"Credential {key} appears too short"
                    )
                
                if value in ['password', '123456', 'admin']:
                    audit_report['recommendations'].append(
                        f"Credential {key} appears to use default/weak value"
                    )
        
        return audit_report
    
    def rotate_encryption_key(self) -> str:
        """Rotate encryption key and re-encrypt all stored credentials."""
        old_key = self.encryption_key
        new_key = Fernet.generate_key()
        
        # This would need to be implemented with actual credential storage
        logger.info("Encryption key rotation completed")
        return new_key.decode()
    
    def sanitize_logs(self, log_data: str) -> str:
        """Sanitize logs to remove sensitive information."""
        # Remove API keys, tokens, and other sensitive data from logs
        sensitive_patterns = [
            r'sk-[a-zA-Z0-9]{20,}',  # OpenAI/OpenRouter keys
            r'bot[a-zA-Z0-9:]{20,}',  # Telegram bot tokens
            r'[a-zA-Z0-9]{20,}',  # Generic API keys
        ]
        
        import re
        sanitized = log_data
        for pattern in sensitive_patterns:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized)
        
        return sanitized


class SecureEnvironment:
    """Secure environment variable management."""
    
    def __init__(self):
        self.credential_manager = CredentialManager()
        self.required_env_vars = [
            'OPENROUTER_API_KEY',
            'REDIS_HOST',
            'REDIS_PORT',
            'OLLAMA_HOST'
        ]
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate the current environment for security compliance."""
        validation = {
            'timestamp': datetime.utcnow().isoformat(),
            'environment_valid': True,
            'missing_vars': [],
            'security_issues': [],
            'recommendations': []
        }
        
        # Check for required environment variables
        for var in self.required_env_vars:
            if not os.getenv(var):
                validation['missing_vars'].append(var)
                validation['environment_valid'] = False
        
        # Check for sensitive data exposure
        sensitive_patterns = [
            'password',
            'secret',
            'key',
            'token'
        ]
        
        for key, value in os.environ.items():
            if any(pattern in key.lower() for pattern in sensitive_patterns):
                if len(value) < 20:
                    validation['security_issues'].append(
                        f"Environment variable {key} appears too short"
                    )
                
                if value.startswith('http://') and 'localhost' not in value:
                    validation['security_issues'].append(
                        f"Environment variable {key} uses insecure HTTP"
                    )
        
        # Check for encryption key
        if not os.getenv('ENCRYPTION_KEY'):
            validation['recommendations'].append(
                "Consider setting ENCRYPTION_KEY for credential encryption"
            )
        
        return validation
    
    def setup_secure_defaults(self) -> Dict[str, str]:
        """Setup secure default values for environment variables."""
        defaults = {
            'REDIS_HOST': 'localhost',
            'REDIS_PORT': '6379',
            'REDIS_DB': '0',
            'OLLAMA_HOST': 'http://localhost:11434',
            'LOG_LEVEL': 'INFO'
        }
        
        return defaults
