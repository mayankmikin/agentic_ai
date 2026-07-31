# Design Summary

*2026-07-09T10:44:57Z*

---

# Market Data API Service Design Summary

This summary outlines the architecture for the Market Data API, focusing on a robust, high-performance, and thread-safe design capable of managing concurrent real-time market data lookups with integrated caching.

## 📦 Components

| Component Name | Purpose | Public Methods |
| :--- | :--- | :--- |
| **MarketDataController** | Entry point for REST requests. Handles request validation, security checks (NFR-04), and delegates the core logic to the Service Layer. | `GET /api/v1/marketdata/{symbol}` |
| **MarketDataService** | Orchestrates the data retrieval flow. Decides whether to fetch from cache or external source, handling the TTL logic and concurrent access control. | `getLivePrice(symbol: String) -> PriceData` |
| **PricingCache** | Manages the storage and retrieval of cached pricing data. Enforces the 60-second TTL (FR-03) and ensures thread safety (NFR-02, C-02). | `get(symbol: String) -> CacheEntry` <br> `set(symbol: String, data: PriceData)` <br> `invalidate(symbol: String)` |
| **ExternalDataProviderAdapter** | The dedicated interface for communicating with the third-party market data provider. Handles API calls, connection pooling, and external error translation (FR-06). | `fetchPrice(symbol: String) -> PriceData` |

***

## ⚙️ Service Layer (Orchestration Sequence)

The MarketDataService orchestrates the following sequence to ensure performance, concurrency, and cache integrity:

**Sequence: `getLivePrice(symbol)`**

1. **[Client Request]** $\rightarrow$ **MarketDataController**: Receives the request and performs Authentication/Authorization checks (NFR-04).
2. **[Controller]** $\rightarrow$ **MarketDataService**: Passes the `symbol` to the service layer.
3. **[Service]** $\rightarrow$ **PricingCache**: Checks for the existence and validity of the entry for the given `symbol`.
    * **Cache Hit (Valid):** If the entry exists and TTL is $\leq$ 60 seconds (FR-03), the cached `PriceData` is immediately returned (NFR-01 performance target met). **[End Sequence]**
    * **Cache Hit (Expired) OR Cache Miss:** The Service proceeds to the Data Fetching phase.
4. **[Service]** $\rightarrow$ **PricingCache**: *Acquires a specific lock* for the target `symbol` to prevent the "thundering herd" problem (FR-05, NFR-02).
5. **[Service]** $\rightarrow$ **ExternalDataProviderAdapter**: Executes `fetchPrice(symbol)`.
6. **[Adapter]** $\rightarrow$ **Service**: Receives the live `PriceData`.
7. **[Service]** $\rightarrow$ **PricingCache**: Stores the newly retrieved data, setting the 60-second TTL (FR-04).
8. **[Service]** $\rightarrow$ **PricingCache**: *Releases the lock* for the target `symbol`.
9. **[Service]** $\rightarrow$ **Controller**: Returns the final `PriceData` object.
10. **[Controller]** $\rightarrow$ **Client**: Returns HTTP 200 OK (or appropriate error code, FR-06).

***

## 🔗 Dependencies

| Type | Dependency | Purpose / Constraint Addressed | Coupling |
| :--- | :--- | :--- | :--- |
| **External** | **Third-Party Market Data API** | Primary source of live price data (FR-01). | High (via Adapter) |
| **External** | **Authentication/Security Provider** | Handles token validation and authorization checks (NFR-04). | High (via Controller) |
| **Internal** | **PricingCache** | Holds state and data. Requires implementation of concurrent maps/mechanisms (e.g., `ConcurrentHashMap` or Read/Write Locks) to ensure thread safety (NFR-02, C-02). | Tight (Service Layer) |
| **Internal** | **Data Transfer Objects (DTOs)** | Structuring `PriceData` and `CacheEntry`. | Low (Interface definitions) |
| **Internal** | **Locking Mechanism** | Must be utilized within the `PricingCache` and `MarketDataService` during refresh to ensure only one thread fetches data for a specific symbol at a time (FR-05, C-02). | Tight (Internal Logic) |