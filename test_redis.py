"""Test Redis Stack with RediSearch integration."""

import redis
import numpy as np
from redis.commands.search.field import VectorField, TextField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# Connect to Redis Stack
r = redis.Redis(host='localhost', port=6380, db=0)

# Test basic Redis connection
try:
    print("Testing Redis connection...")
    print(f"Ping: {r.ping()}")
    
    # Check if RediSearch is available
    index_name = "test_index"
    try:
        # Define schema
        schema = (
            VectorField("embedding", "FLAT", {
                "TYPE": "FLOAT32",
                "DIM": 1536,  # Dimension for the test model
                "DISTANCE_METRIC": "COSINE"
            }),
            TextField("text"),
            NumericField("score")
        )
        
        # Create the index
        definition = IndexDefinition(prefix=["doc:"])
        r.ft(index_name).create_index(fields=schema, definition=definition)
        print("✅ Created test RediSearch index")
        
        # Test storing a document
        test_embedding = np.random.rand(1536).astype(np.float32).tobytes()
        doc_id = "doc:test1"
        r.hset(doc_id, mapping={
            "embedding": test_embedding,
            "text": "test document",
            "score": 1.0
        })
        
        print("✅ Successfully stored test document with vector")
        
    except Exception as e:
        print(f"❌ RediSearch test failed: {e}")
    finally:
        # Clean up
        print("Cleaning up test data...")
        try:
            r.ft(index_name).dropindex(delete_documents=True)
            print("✅ Dropped test index.")
        except redis.exceptions.ResponseError:
            print("Test index did not exist, no cleanup needed.")
        except Exception as e:
            print(f"Could not clean up index: {e}")

except Exception as e:
    print(f"❌ Redis connection failed: {e}")

print("Test completed.")
