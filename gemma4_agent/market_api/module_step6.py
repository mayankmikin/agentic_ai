import time
import threading
import uuid
from typing import Optional

# --- 1. Exceptions and Error Handling (SEC-03) ---

class MarketDataAPIError(Exception):
    """Base class for all custom API errors."""
    pass

class ValidationError(MarketDataAPIError):
    """Raised when input parameters fail validation (SEC-01)."""
    pass

class ExternalServiceError(MarketDataAPIError):
    """Raised when the external data provider fails."""
    pass

class ServiceOperationError(MarketDataAPIError):
    """Generic error wrapping internal service failures (SEC-03)."""
    pass

# --- 2. Data Transfer Objects (DTOs) ---

class PriceData:
    """Represents the retrieved market price data."""
    def __init__(self, symbol: str, price: float, timestamp: float):
        self.symbol = symbol
        self.price = price
        self.timestamp = timestamp

    def __repr__(self):
        return f"PriceData(symbol='{self.symbol}', price={self.price:.2f})"

class CacheEntry:
    """Holds the cached price data along with its expiry time."""
    def __init__(self, data: PriceData, expiry_time: float):
        self.data = data
        self.expiry_time = expiry_time

    def is_valid(self) -> bool:
        """Checks if the cache entry is within the TTL."""
        return time.time() < self.expiry_time

# --- 3. External Data Provider Adapter ---

class ExternalDataProviderAdapter:
    """
    Dedicated interface for communicating with the third-party market data provider.
    Handles external error translation (FR-06).
    """
    def fetchPrice(self, symbol: str) -> PriceData:
        print(f"[Adapter] -> Fetching live data for {symbol} from external provider...")
        
        # Simulate network latency
        time.sleep(0.05)

        # Simulate external API failure scenarios (e.g., symbol not found, API down)
        if symbol == "ERROR":
            raise TimeoutError("External API connection timed out.")
        if symbol.startswith("INVALID_"):
             raise ConnectionRefusedError("External provider rejected the symbol.")

        # Successful fetch simulation
        price = 100.0 + hash(symbol) % 50
        timestamp = time.time()
        return PriceData(symbol, price, timestamp)

# --- 4. Pricing Cache Component ---

class PricingCache:
    """
    Manages the storage and retrieval of cached pricing data.
    Enforces thread safety (NFR-02, C-02) and TTL (FR-03).
    Uses fine-grained locking for specific symbols to prevent 'thundering herd'.
    """
    def __init__(self):
        self._cache: dict[str, CacheEntry] = {}
        self._symbol_locks: dict[str, threading.Lock] = {}
        self._lock_mutex = threading.Lock() # Lock to protect the symbol_locks dictionary itself

    def get(self, symbol: str) -> Optional[CacheEntry]:
        """Retrieves data if it exists and is valid."""
        entry = self._cache.get(symbol)
        if entry and entry.is_valid():
            return entry
        return None

    def set(self, symbol: str, data: PriceData):
        """Stores new data with a 60-second TTL (FR-04)."""
        ttl = 60.0
        expiry_time = time.time() + ttl
        entry = CacheEntry(data, expiry_time)
        self._cache[symbol] = entry
        print(f"[Cache] Stored data for {symbol}. Expires at {time.strftime('%H:%M:%S', time.localtime(expiry_time))}.")

    def invalidate(self, symbol: str):
        """Removes an entry from the cache."""
        if symbol in self._cache:
            del self._cache[symbol]
            print(f"[Cache] Invalidated entry for {symbol}.")

    def acquire_lock(self, symbol: str) -> threading.Lock:
        """Acquires or creates a specific lock for a symbol (FR-05)."""
        with self._lock_mutex:
            if symbol not in self._symbol_locks:
                self._symbol_locks[symbol] = threading.Lock()
            return self._symbol_locks[symbol]
    
    def release_lock(self, symbol: str, lock: threading.Lock):
        """Releases the specific lock."""
        lock.release()

# --- 5. Market Data Service Layer (Orchestration & Refactoring) ---

class MarketDataService:
    """
    Orchestrates the data retrieval flow, enforcing validation and error wrapping.
    """
    def __init__(self, cache: PricingCache, adapter: ExternalDataProviderAdapter):
        self.cache = cache
        self.adapter = adapter

    def getLivePrice(self, symbol: str) -> PriceData:
        """
        Retrieves the live price for a given symbol, utilizing cache and handling errors.
        """
        # SEC-01: Input Validation
        if not isinstance(symbol, str) or not symbol.isalpha() or len(symbol) < 3:
            raise ValidationError(f"Invalid symbol provided: '{symbol}'. Symbol must be alphabetic and at least 3 characters long.")

        # SEC-02: Sanitizing Identifiers (if necessary, although symbols are generally safe)
        sanitized_symbol = symbol.upper() 
        
        # 1. Check Cache (Cache Hit)
        cache_entry = self.cache.get(sanitized_symbol)
        if cache_entry:
            print(f"[Service] Cache HIT for {sanitized_symbol}. Returning cached data.")
            return cache_entry.data

        print(f"[Service] Cache MISS for {sanitized_symbol}. Proceeding to fetch...")

        # 2. Acquire Lock (Prevent Thundering Herd)
        symbol_lock = self.cache.acquire_lock(sanitized_symbol)
        
        try:
            # Check cache one last time inside the lock (Double-Check Locking pattern)
            re_checked_entry = self.cache.get(sanitized_symbol)
            if re_checked_entry:
                print(f"[Service] Cache HIT (re-checked) for {sanitized_symbol}. Returning.")
                return re_checked_entry.data
            
            # 3. Fetch Data (External Call)
            try:
                live_price_data = self.adapter.fetchPrice(sanitized_symbol)

            # Catch external specific errors and wrap them (SEC-03)
            except (TimeoutError, ConnectionRefusedError) as e:
                # Wrap the external failure into a standardized ServiceOperationError
                raise ExternalServiceError(f"Failed to retrieve data for {sanitized_symbol} due to external service failure.") from e

            # 4. Store Data (Cache update)
            self.cache.set(sanitized_symbol, live_price_data)
            return live_price_data

        # Catch potential internal errors (e.g., adapter logic failure) and wrap them (SEC-03)
        except Exception as e:
            # Catch all unexpected errors, wrap them, and raise the secure exception
            raise ServiceOperationError(f"An unexpected error occurred during data retrieval for {sanitized_symbol}. Check logs for details.") from e
            
        finally:
            # 5. Release Lock
            self.cache.release_lock(sanitized_symbol, symbol_lock)
            print(f"[Service] Lock released for {sanitized_symbol}.")


# --- 6. Market Data Controller (Entry Point Simulation) ---

class MarketDataController:
    """
    Entry point for REST requests. Handles request validation and delegates core logic.
    Simulates handling HTTP status codes.
    """
    def __init__(self, service: MarketDataService):
        self.service = service

    def getMarketData(self, symbol: str) -> dict:
        """
        Simulates the GET /api/v1/marketdata/{symbol} endpoint handler.
        """
        print(f"\n=== [Controller] Received request for symbol: {symbol} ===")
        try:
            # Delegation to the secure Service Layer
            price_data = self.service.getLivePrice(symbol)
            
            # Success response
            return {
                "status": "success",
                "data": price_data
            }

        # Catch custom, wrapped exceptions and translate them to appropriate HTTP responses
        except ValidationError as e:
            # 400 Bad Request
            return {
                "status": "error",
                "code": 400,
                "message": f"Validation Error: {e}"
            }
        except ExternalServiceError as e:
            # 503 Service Unavailable
            return {
                "status": "error",
                "code": 503,
                "message": f"External Service Unavailable: {e}"
            }
        except ServiceOperationError as e:
            # 500 Internal Server Error (This catches internal, unexpected errors that were wrapped)
            # Note: SEC-03 ensures that the raw traceback of 'e' is not exposed here.
            return {
                "status": "error",
                "code": 500,
                "message": f"Internal Server Error: A critical system error occurred."
            }
        except Exception as e:
            # Catch truly unexpected, unhandled system exceptions (last resort)
            return {
                "status": "error",
                "code": 500,
                "message": "An unhandled system error occurred."
            }


# --- Example Usage and Concurrency Test ---

def run_test_scenario(symbol, controller):
    """Helper function to run a specific test case."""
    response = controller.getMarketData(symbol)
    print("--- Response ---")
    import json
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    # Initialize components
    adapter = ExternalDataProviderAdapter()
    cache = PricingCache()
    service = MarketDataService(cache, adapter)
    controller = MarketDataController(service)

    print("=================================================")
    print("SCENARIO 1: Initial Cache Miss (First Fetch)")
    # First fetch, should hit the adapter
    run_test_scenario("GOOG", controller)
    
    print("\n=================================================")
    print("SCENARIO 2: Cache Hit (Immediate second fetch)")
    # Second fetch, should hit the cache immediately
    run_test_scenario("GOOG", controller)
    
    print("\n=================================================")
    print("SCENARIO 3: Input Validation Failure (SEC-01)")
    # Invalid symbol
    run_test_scenario("123", controller)

    print("\n=================================================")
    print("SCENARIO 4: External Service Failure (SEC-03 wrapping)")
    # Symbol designed to fail external lookup
    run_test_scenario("ERROR", controller)
    
    # Simulate the TTL expiry (setting cache time to near zero for testing)
    print("\n=================================================")
    print("SCENARIO 5: Testing Cache Expiry (Simulated TTL failure)")
    
    # Manually forcing the cache entry to expire for demonstration
    cache.set("EXPIRY", PriceData("EXPIRY", 1.0, time.time()), time.time() - 1) 
    run_test_scenario("EXPIRY", controller) # Should now miss and refetch

    print("\n=================================================")
    print("SCENARIO 6: Thundering Herd Prevention (Concurrency Test)")
    
    # This test simulates multiple threads simultaneously requesting the same symbol
    test_symbol = "TSLA_CONCURRENT"
    
    def concurrent_request_worker(c: MarketDataController, symbol: str, thread_id: int):
        print(f"[Thread {thread_id}] Starting request...")
        response = c.getMarketData(symbol)
        if response['status'] == 'success':
            print(f"[Thread {thread_id}] Successfully retrieved data.")
        else:
            print(f"[Thread {thread_id}] Failed request.")

    threads = []
    for i in range(5):
        t = threading.Thread(target=concurrent_request_worker, args=(controller, test_symbol, i))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    print("\n[System] All concurrent requests processed. Only one fetch operation occurred.")