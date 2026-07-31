import time
import threading
from typing import Optional, Dict, Any

class PriceData:
    """DTO representing the market price information."""
    def __init__(self, symbol: str, price: float, volume: int):
        self.symbol = symbol
        self.price = price
        self.volume = volume

    def __repr__(self):
        return f"PriceData(symbol={self.symbol}, price={self.price}, volume={self.volume})"

class CacheEntry:
    """DTO storing cached data along with its timestamp."""
    def __init__(self, data: PriceData, timestamp: float):
        self.data = data
        self.timestamp = timestamp

    def is_valid(self, ttl_seconds: float = 60.0) -> bool:
        """Checks if the entry is within the Time To Live window."""
        return (time.time() - self.timestamp) <= ttl_seconds

class CacheKeyError(Exception):
    """Custom exception for when a key is invalid or missing in the cache."""
    pass

class PricingCache:
    """
    Manages the storage and retrieval of cached pricing data, ensuring thread safety
    and implementing per-symbol locking to prevent the 'thundering herd' problem.
    """
    def __init__(self, ttl_seconds: float = 60.0):
        # Main storage for the cache: symbol -> CacheEntry
        self._cache: Dict[str, CacheEntry] = {}
        # Global lock for operations on the cache dictionary itself (e.g., adding/removing keys)
        self._cache_lock = threading.Lock()
        # Per-symbol locks to ensure only one thread refreshes a specific symbol at a time
        self._symbol_locks: Dict[str, threading.Lock] = {}
        self.ttl_seconds = ttl_seconds

    def _get_symbol_lock(self, symbol: str) -> threading.Lock:
        """Retrieves or creates a specific lock for a given symbol."""
        with self._cache_lock:
            if symbol not in self._symbol_locks:
                self._symbol_locks[symbol] = threading.Lock()
            return self._symbol_locks[symbol]

    def get(self, symbol: str) -> PriceData:
        """
        Retrieves data from the cache if it exists and is not expired.

        Raises CacheKeyError if the symbol is not found or the entry is expired.
        """
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Symbol must be a non-empty string.")

        with self._cache_lock:
            if symbol not in self._cache:
                raise CacheKeyError(f"Symbol '{symbol}' not found in cache.")
            
            entry = self._cache[symbol]

        if not entry.is_valid(self.ttl_seconds):
            # Logically expired, caller must refresh
            raise CacheKeyError(f"Symbol '{symbol}' entry has expired.")

        return entry.data

    def set(self, symbol: str, data: PriceData):
        """
        Stores new data into the cache, setting the current timestamp.
        """
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Symbol must be a non-empty string.")
        
        entry = CacheEntry(data, time.time())

        with self._cache_lock:
            self._cache[symbol] = entry

    def invalidate(self, symbol: str):
        """Removes an entry from the cache, forcing a refresh on the next request."""
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Symbol must be a non-empty string.")
            
        with self._cache_lock:
            if symbol in self._cache:
                del self._cache[symbol]

    def acquire_symbol_lock(self, symbol: str) -> threading.Lock:
        """
        Acquires the specific per-symbol lock. This is used by the Service Layer
        to prevent multiple concurrent fetching operations for the same symbol.
        """
        # The lock object is obtained and returned, allowing the caller to handle the context manager.
        return self._get_symbol_lock(symbol)

# Example Usage (Self-Test)
if __name__ == '__main__':
    print("--- PricingCache Self-Test ---")
    cache = PricingCache(ttl_seconds=5)
    
    test_symbol = "AAPL"
    
    # 1. Set Data
    data1 = PriceData(test_symbol, 150.0, 1000)
    cache.set(test_symbol, data1)
    print(f"Cache set for {test_symbol}.")

    # 2. Cache Hit (Valid)
    try:
        price_data = cache.get(test_symbol)
        print(f"Cache GET SUCCESS: {price_data}")
    except CacheKeyError as e:
        print(f"Cache GET FAIL: {e}")
        
    # 3. Test Expiration (Wait 6 seconds)
    print("\nWaiting 6 seconds to trigger TTL expiry...")
    time.sleep(6)
    
    # 4. Cache Hit (Expired) -> Should throw CacheKeyError
    try:
        cache.get(test_symbol)
    except CacheKeyError as e:
        print(f"Cache GET EXPIRED SUCCESS: {e}")
        
    # 5. Invalidation
    cache.invalidate(test_symbol)
    print(f"Cache invalidated for {test_symbol}.")
    
    # 6. Cache Miss (Should throw CacheKeyError)
    try:
        cache.get(test_symbol)
    except CacheKeyError as e:
        print(f"Cache GET MISS SUCCESS: {e}")

    # 7. Test Concurrency (Thundering Herd Prevention)
    print("\nTesting concurrent access...")
    
    # Set a new entry
    data_concurrent = PriceData("GOOG", 2500.0, 500)
    cache.set("GOOG", data_concurrent)
    
    symbol_to_test = "GOOG"
    
    def fetch_and_update(symbol):
        # Simulate external API fetching and processing time
        time.sleep(0.1) 
        
        # Acquire lock before simulating fetching
        lock = cache.acquire_symbol_lock(symbol)
        with lock:
            # Only the first thread to acquire this block proceeds with the refresh logic
            try:
                # Try to get the data (will fail if we simulate cache expiry)
                cache.get(symbol) 
                print(f"[Thread {threading.get_ident()}] Cache hit (no refresh needed).")
            except CacheKeyError:
                print(f"[Thread {threading.get_ident()}] Cache Miss/Expired. Simulating API Fetching...")
                # Simulate external call
                new_data = PriceData(symbol, 2501.0, 600)
                # Update cache
                cache.set(symbol, new_data)
                print(f"[Thread {threading.get_ident()}] Refresh COMPLETE. New Price: {new_data.price}")

    threads = []
    for i in range(5):
        t = threading.Thread(target=fetch_and_update, args=(symbol_to_test,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    
    print("\nConcurrency test finished. Only one thread should report 'Refresh COMPLETE'.")
    
    # 8. Test Input Validation
    try:
        cache.get("")
    except ValueError as e:
        print(f"Input Validation SUCCESS: Caught expected error: {e}")
    
    try:
        cache.set("AAPL", PriceData("AAPL", 100, 10))
        cache.get(None)
    except TypeError:
        print("Input Validation SUCCESS: Caught TypeError on invalid symbol.")<unused56>