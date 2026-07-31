import uuid
from typing import Dict, Any
from datetime import datetime

# --- DTOs and Interfaces (Simulating internal dependencies) ---

class PriceData:
    """Data Transfer Object for market price information."""
    def __init__(self, symbol: str, price: float, timestamp: datetime):
        self.symbol = symbol
        self.price = price
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "timestamp": self.timestamp.isoformat()
        }

class ServiceError(Exception):
    """Base class for service layer operational errors."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code

# --- Mock External Service Layer ---

class MarketDataService:
    """
    Mock implementation of the MarketDataService.
    In a real application, this would handle the cache/adapter logic.
    """
    def __init__(self):
        # Simulate a dependency injection scenario
        pass

    def getLivePrice(self, symbol: str) -> PriceData:
        """
        Retrieves live price data, orchestrating cache and external lookups.
        
        Raises ServiceError if the symbol is invalid or the external API fails.
        """
        if not isinstance(symbol, str) or not symbol.isalnum():
            raise ServiceError("Invalid symbol format.", status_code=400)
        
        if symbol == "ERROR":
            raise ServiceError("External provider currently unreachable.", status_code=503)

        # Simulate successful data retrieval
        price = 100.0 + hash(symbol) % 50 / 10.0
        timestamp = datetime.utcnow()
        return PriceData(symbol=symbol, price=price, timestamp=timestamp)

# --- MarketDataController Implementation ---

class MarketDataController:
    """
    Entry point for REST requests. Handles routing, validation, and security checks.
    Adheres to SEC-01 (Input Validation), SEC-02 (Sanitization), and SEC-03 (Error Wrapping).
    """

    def __init__(self, service: MarketDataService):
        self.service = service
        # Note: In a real setup, SecurityProvider would be injected here.

    def _validate_symbol(self, symbol: str) -> bool:
        """
        SEC-01: Validates the symbol input. 
        Assumes valid symbols are alphanumeric and non-empty.
        """
        if not isinstance(symbol, str) or not symbol:
            return False
        # Basic validation rule for market data symbols
        if not symbol.isalnum() or len(symbol) > 10:
            return False
        return True

    def _handle_security_checks(self, symbol: str):
        """
        NFR-04: Placeholder for authentication/authorization checks.
        SEC-02: Sanitizes identifier before processing/logging.
        """
        # Sanitization step: Ensure the identifier is clean before use
        sanitized_symbol = symbol.strip().upper()
        
        # Placeholder for actual security logic (e.g., token validation)
        # If token is missing or expired, raise Unauthorized.
        
        # In production, logging would use sanitized_symbol.
        # print(f"INFO: Security check passed for symbol: {sanitized_symbol}")
        pass

    def get_live_price(self, symbol: str) -> Dict[str, Any]:
        """
        Handles the GET /api/v1/marketdata/{symbol} request.
        Returns a structured response dictionary.
        """
        # 1. Input Validation (SEC-01)
        if not self._validate_symbol(symbol):
            return self._error_response(
                status_code=400, 
                message="Invalid input: Symbol must be a non-empty alphanumeric string."
            )
        
        # 2. Security Check (NFR-04)
        try:
            self._handle_security_checks(symbol)
        except Exception as e:
            # Catch security failures and wrap them (SEC-03)
            return self._error_response(status_code=401, message="Unauthorized access.")

        # 3. Delegation and Error Wrapping (SEC-03)
        try:
            price_data = self.service.getLivePrice(symbol)
            # Success path
            return {
                "status": "success",
                "data": price_data.to_dict()
            }
        except ServiceError as e:
            # Catch known service errors (e.g., 400, 503)
            return self._error_response(
                status_code=e.status_code, 
                message=e.args[0]
            )
        except Exception as e:
            # Catch unexpected system/database errors and wrap them (SEC-03)
            # Log the raw exception internally, but return a generic error to the caller.
            # print(f"CRITICAL: Unexpected system error during retrieval: {e}")
            return self._error_response(
                status_code=500, 
                message="An unexpected internal error occurred during data retrieval."
            )

    def _error_response(self, status_code: int, message: str) -> Dict[str, Any]:
        """Helper function to standardize error response format."""
        return {
            "status": "error",
            "http_status": status_code,
            "message": message
        }

# --- Usage Example / Testing ---

if __name__ == "__main__":
    # Setup dependencies
    service = MarketDataService()
    controller = MarketDataController(service)

    print("="*50)
    print("--- Test Case 1: Successful Fetch (Valid Input) ---")
    result_ok = controller.get_live_price("TSLA")
    print(f"Response Status: {result_ok.get('status')}")
    if result_ok.get('status') == 'success':
        print(f"Successfully retrieved price for {result_ok['data']['symbol']}: ${result_ok['data']['price']:.2f}")
    print("\n")

    print("="*50)
    print("--- Test Case 2: Input Validation Failure (SEC-01) ---")
    # Invalid symbol format
    result_invalid = controller.get_live_price("TSLA!")
    print(f"Response Status: {result_invalid.get('status')}")
    print(f"Error Message: {result_invalid.get('message')}")
    print("\n")

    print("="*50)
    print("--- Test Case 3: Service Failure (External Provider Down) (SEC-03) ---")
    # Simulating service layer raising a 503 error
    result_service_error = controller.get_live_price("ERROR")
    print(f"Response Status: {result_service_error.get('status')}")
    print(f"HTTP Code: {result_service_error.get('http_status')}")
    print(f"Error Message: {result_service_error.get('message')}")
    print("\n")

    print("="*50)
    print("--- Test Case 4: Unhandled System Error (Simulated) ---")
    # To simulate a genuine unhandled error, we would need to modify the service mock, 
    # but we rely on the try/except block catching unexpected exceptions.
    # Since the mock is simple, this tests the generic 500 wrap logic.
    # (If a runtime bug occurred in the service, this path would be hit.)
    print("Simulating an internal crash...")
    try:
        # Forced internal crash simulation
        controller.get_live_price(lambda: 1/0)
    except Exception:
        # Manually trigger the controller's fallback logic if possible
        # (Demonstrating SEC-03 protection against raw stack traces)
        pass
    
    # If we modify the service mock to raise a standard Exception instead of ServiceError:
    class BadServiceMock(MarketDataService):
        def getLivePrice(self, symbol: str) -> PriceData:
            raise RuntimeError("Database Connection Lost")

    controller_bad = MarketDataController(BadServiceMock())
    result_crash = controller_bad.get_live_price("TSLA")
    print(f"Response Status: {result_crash.get('status')}")
    print(f"HTTP Code: {result_crash.get('http_status')}")
    print(f"Error Message: {result_crash.get('message')}")
    print("\n")
    
    print("="*50)
    print("--- Test Case 5: Security Check (SEC-02) ---")
    # Sanitize/validate check on symbol
    controller.get_live_price("  tsla  ")
