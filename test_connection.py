#!/usr/bin/env python3
"""
Test script to verify MovieMate API connection
"""

import requests
import json
import time

def test_api_connection():
    """Test the MovieMate API endpoints"""
    
    base_url = "http://localhost:5000"
    
    print("🧪 Testing MovieMate API Connection")
    print("=" * 40)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Status: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")
        print("   Make sure the server is running on port 5000")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Root endpoint passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test 3: Recommendation endpoint
    print("\n3. Testing recommendation endpoint...")
    test_data = {
        "movie_title": "The Matrix",
        "user_id": 1,
        "num_recommendations": 3
    }
    
    try:
        response = requests.post(
            f"{base_url}/recommend", 
            json=test_data, 
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if "recommendations" in data:
                print("✅ Recommendation endpoint passed")
                print(f"   Found {len(data['recommendations'])} recommendations")
                for i, rec in enumerate(data['recommendations'][:2], 1):
                    print(f"   {i}. {rec['title']} ({rec['match_percentage']:.1f}%)")
            else:
                print("❌ No recommendations in response")
                print(f"   Response: {data}")
        else:
            print(f"❌ Recommendation endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Recommendation endpoint error: {e}")
    
    print("\n🎬 MovieMate API Test Complete!")
    return True

if __name__ == "__main__":
    test_api_connection() 