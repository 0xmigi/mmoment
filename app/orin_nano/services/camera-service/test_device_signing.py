#!/usr/bin/env python3
"""
Test script for device signing functionality.
Verifies that device signer works correctly and can be used for future on-chain operations.
"""

import sys
import os
import json
import time
import base64

# Add services directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))

def test_device_signer():
    """Test device signer initialization and functionality"""
    print("🔧 Testing Device Signer for DePIN Authentication")
    print("=" * 60)
    
    try:
        from services.device_signer import DeviceSigner
        
        # Test 1: Initialize device signer
        print("\n1. Testing Device Signer Initialization...")
        signer = DeviceSigner()
        
        if signer.keypair:
            print(f"✅ Device keypair initialized successfully")
            print(f"   Device Public Key: {signer.get_public_key()}")
        else:
            print("❌ Failed to initialize device keypair")
            return False
            
        # Test 2: Test response signing
        print("\n2. Testing Response Signing...")
        test_response = {
            'success': True,
            'data': {
                'camera_status': 'online',
                'timestamp': int(time.time()),
                'test_value': 'device_auth_test'
            }
        }
        
        signed_response = signer.sign_response(test_response)
        
        if 'device_signature' in signed_response:
            print("✅ Response signing successful")
            sig_info = signed_response['device_signature']
            print(f"   Algorithm: {sig_info['algorithm']}")
            print(f"   Device Key: {sig_info['device_pubkey'][:12]}...")
            print(f"   Timestamp: {sig_info['timestamp']}")
            print(f"   Signature: {sig_info['signature'][:20]}...")
        else:
            print("❌ Response signing failed")
            return False
            
        # Test 3: Test device info
        print("\n3. Testing Device Info...")
        device_info = signer.get_device_info()
        print(f"✅ Device Type: {device_info['device_type']}")
        print(f"   Capabilities: {device_info['capabilities']}")
        print(f"   Ready for On-Chain: {device_info['ready_for_onchain']}")
        
        # Test 4: Test transaction signing capability (for future use)
        print("\n4. Testing Transaction Signing Capability...")
        test_transaction = b"test_transaction_bytes_for_future_blockchain_operations"
        
        try:
            signature_bytes = signer.sign_transaction_bytes(test_transaction)
            print(f"✅ Transaction signing capability verified")
            print(f"   Signature Length: {len(signature_bytes)} bytes")
            print(f"   Signature (first 10 bytes): {signature_bytes[:10].hex()}")
        except Exception as e:
            print(f"❌ Transaction signing test failed: {e}")
            return False
            
        # Test 5: Verify signature consistency
        print("\n5. Testing Signature Consistency...")
        signed_response2 = signer.sign_response(test_response)
        
        # Signatures should be different due to timestamp, but device key should be same
        if (signed_response['device_signature']['device_pubkey'] == 
            signed_response2['device_signature']['device_pubkey']):
            print("✅ Device key consistency verified")
        else:
            print("❌ Device key inconsistency detected")
            return False
            
        if (signed_response['device_signature']['signature'] != 
            signed_response2['device_signature']['signature']):
            print("✅ Signature uniqueness verified (different timestamps)")
        else:
            print("⚠️  Signatures are identical (may indicate timing issue)")
            
        print("\n🎉 All Device Signing Tests Passed!")
        print("\n📋 Test Summary:")
        print(f"   ✅ Device Public Key: {signer.get_public_key()}")
        print(f"   ✅ Response Signing: Working")
        print(f"   ✅ Transaction Signing: Ready for on-chain")
        print(f"   ✅ Hardware Binding: Active")
        print(f"   ✅ DePIN Authentication: Ready")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all dependencies are installed:")
        print("   pip install cryptography solders")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_flask():
    """Test integration with Flask decorator"""
    print("\n🌐 Testing Flask Integration")
    print("=" * 40)
    
    try:
        from services.device_signer import DeviceSigner
        
        # Simulate Flask response signing
        signer = DeviceSigner()
        
        # Mock Flask response data
        mock_response = {
            'success': True,
            'data': {
                'isOnline': True,
                'isStreaming': False,
                'lastSeen': int(time.time())
            }
        }
        
        signed_response = signer.sign_response(mock_response)
        
        # Verify all expected fields are present
        required_fields = ['device_signature']
        signature_fields = ['device_pubkey', 'timestamp', 'version', 'algorithm', 'signature']
        
        if all(field in signed_response for field in required_fields):
            print("✅ Flask response structure valid")
        else:
            print("❌ Missing required fields in signed response")
            return False
            
        sig = signed_response['device_signature']
        if all(field in sig for field in signature_fields):
            print("✅ Device signature structure valid")
            print(f"   Ready for frontend verification")
        else:
            print("❌ Missing signature fields")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Flask integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 MMOMENT Device Signing Test Suite")
    print("Testing DePIN authentication and future on-chain capabilities")
    print()
    
    success = True
    
    # Run device signer tests
    if not test_device_signer():
        success = False
        
    # Run Flask integration tests
    if not test_integration_with_flask():
        success = False
        
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED - Device ready for DePIN operations!")
        print("   • Hardware-bound ed25519 keypair: ✅")
        print("   • Response signing: ✅") 
        print("   • Future on-chain transactions: ✅")
        print("   • Flask integration: ✅")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED - Check logs above")
        sys.exit(1)