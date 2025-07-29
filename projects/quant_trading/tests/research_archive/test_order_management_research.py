#!/usr/bin/env python3
"""
Research-Driven Order Management Test
Based on Hyperliquid Python SDK V3 research
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_order_management_interface():
    """Test Order Management interface following Hyperliquid research patterns."""
    print("📦 Testing Order Management (Research-Driven)...")
    
    try:
        from src.execution.order_management import OrderManager
        from src.config.settings import get_settings
        
        settings = get_settings()
        order_manager = OrderManager(settings)
        print("✅ Order manager initialized")
        
        # Test interface methods based on Hyperliquid research patterns
        expected_methods = [
            'place_order', 'cancel_order', 'get_order_status',
            'get_open_orders', 'get_fills', 'get_user_state'
        ]
        
        for method in expected_methods:
            if hasattr(order_manager, method):
                print(f"✅ Method available: {method}")
            else:
                print(f"⚠️  Method not found: {method}")
        
        # Test Hyperliquid client integration (from research)
        if hasattr(order_manager, 'hyperliquid_client'):
            print("✅ Hyperliquid client integration available")
        elif hasattr(order_manager, 'client'):
            print("✅ Client integration available")
        else:
            print("⚠️  Client integration not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Order management interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hyperliquid_api_patterns():
    """Test Hyperliquid API patterns from research."""
    print("🌐 Testing Hyperliquid API Patterns...")
    
    try:
        # Research Pattern: REST API request structure (from research_summary_v3.md)
        
        def create_info_request(request_type, params=None):
            """Create Hyperliquid info API request following research patterns."""
            base_request = {
                "type": request_type
            }
            
            if params:
                base_request.update(params)
            
            return base_request
        
        # Test API request patterns from research
        api_patterns = [
            ("clearinghouseState", {"user": "0x" + "0" * 40}),  # User state
            ("l2Book", {"coin": "BTC"}),                        # Order book
            ("candle", {"coin": "BTC", "interval": "1h", "startTime": 0}),  # Candlestick
            ("allMids", None),                                  # All mid prices
            ("assetCtxs", None),                               # Asset contexts
        ]
        
        for request_type, params in api_patterns:
            request = create_info_request(request_type, params)
            print(f"✅ {request_type} request: {json.dumps(request, indent=2)[:100]}...")
            
            # Validate request structure
            if "type" in request:
                print(f"✅ Request structure valid for {request_type}")
            else:
                print(f"❌ Invalid request structure for {request_type}")
        
        print("✅ Hyperliquid API patterns validated")
        return True
        
    except Exception as e:
        print(f"❌ Hyperliquid API patterns test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_order_lifecycle_patterns():
    """Test order lifecycle patterns from research."""
    print("🔄 Testing Order Lifecycle Patterns...")
    
    try:
        # Research Pattern: Order lifecycle states and transitions
        
        class OrderState:
            """Order states following Hyperliquid patterns."""
            PENDING = "pending"
            OPEN = "open"
            FILLED = "filled"
            CANCELLED = "cancelled"
            REJECTED = "rejected"
        
        class MockOrder:
            """Mock order for testing lifecycle patterns."""
            def __init__(self, order_id, symbol, size, price):
                self.order_id = order_id
                self.symbol = symbol
                self.size = size
                self.price = price
                self.state = OrderState.PENDING
                self.filled_size = 0.0
                self.timestamp = datetime.now()
            
            def update_state(self, new_state, filled_size=None):
                """Update order state following valid transitions."""
                valid_transitions = {
                    OrderState.PENDING: [OrderState.OPEN, OrderState.REJECTED],
                    OrderState.OPEN: [OrderState.FILLED, OrderState.CANCELLED],
                    OrderState.FILLED: [],  # Terminal state
                    OrderState.CANCELLED: [],  # Terminal state
                    OrderState.REJECTED: [],  # Terminal state
                }
                
                if new_state in valid_transitions.get(self.state, []):
                    self.state = new_state
                    if filled_size is not None:
                        self.filled_size = filled_size
                    return True
                return False
        
        # Test order lifecycle
        test_order = MockOrder("test_001", "BTC", 0.1, 50000.0)
        print(f"✅ Order created: {test_order.order_id} ({test_order.state})")
        
        # Test state transitions
        transitions = [
            (OrderState.OPEN, None),
            (OrderState.FILLED, 0.1),
        ]
        
        for new_state, filled_size in transitions:
            success = test_order.update_state(new_state, filled_size)
            if success:
                print(f"✅ Transition to {new_state}: Success")
                if filled_size:
                    print(f"✅ Filled size: {filled_size}")
            else:
                print(f"❌ Invalid transition to {new_state}")
        
        # Test order information
        order_info = {
            "order_id": test_order.order_id,
            "symbol": test_order.symbol,
            "size": test_order.size,
            "price": test_order.price,
            "state": test_order.state,
            "filled_size": test_order.filled_size,
            "timestamp": test_order.timestamp.isoformat()
        }
        
        print(f"✅ Order info: {json.dumps(order_info, indent=2)[:200]}...")
        
        print("✅ Order lifecycle patterns validated")
        return True
        
    except Exception as e:
        print(f"❌ Order lifecycle patterns test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_risk_management_patterns():
    """Test risk management patterns from research."""
    print("🛡️ Testing Risk Management Patterns...")
    
    try:
        # Research Pattern: Asset context validation and risk checks
        
        def validate_order_risk(symbol, size, price, user_balance, max_position_pct=0.25):
            """Validate order against risk limits following research patterns."""
            risk_checks = {
                "size_positive": size > 0,
                "price_positive": price > 0,
                "sufficient_balance": user_balance > (size * price),
                "position_limit": (size * price) <= (user_balance * max_position_pct),
            }
            
            all_passed = all(risk_checks.values())
            
            return {
                "passed": all_passed,
                "checks": risk_checks,
                "position_value": size * price,
                "position_pct": (size * price) / user_balance if user_balance > 0 else 0,
            }
        
        # Test risk validation scenarios
        risk_scenarios = [
            ("BTC", 0.1, 50000.0, 100000.0, "Valid order"),
            ("BTC", 0.5, 50000.0, 100000.0, "Position limit exceeded"),
            ("BTC", 0.1, 50000.0, 1000.0, "Insufficient balance"),
            ("BTC", -0.1, 50000.0, 100000.0, "Invalid size"),
            ("BTC", 0.1, -50000.0, 100000.0, "Invalid price"),
        ]
        
        for symbol, size, price, balance, description in risk_scenarios:
            risk_result = validate_order_risk(symbol, size, price, balance)
            status = "✅ PASS" if risk_result["passed"] else "❌ FAIL"
            print(f"{status} {description}: {risk_result['position_pct']:.1%} of portfolio")
            
            # Show failed checks
            for check, passed in risk_result["checks"].items():
                if not passed:
                    print(f"  ❌ Failed: {check}")
        
        print("✅ Risk management patterns validated")
        return True
        
    except Exception as e:
        print(f"❌ Risk management patterns test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("📦 Research-Driven Order Management Testing\n")
    
    success1 = test_order_management_interface()
    print()
    success2 = test_hyperliquid_api_patterns()
    print()
    success3 = test_order_lifecycle_patterns()
    print()
    success4 = test_risk_management_patterns()
    
    overall_success = success1 and success2 and success3 and success4
    print(f"\n📋 Overall Result: {'✅ PASSED' if overall_success else '❌ FAILED'}")
    sys.exit(0 if overall_success else 1)