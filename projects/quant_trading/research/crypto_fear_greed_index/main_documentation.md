# Crypto Fear & Greed Index - Complete Documentation

**Source URL**: https://alternative.me/crypto/fear-and-greed-index/  
**Extraction Method**: Brightdata MCP  
**Extraction Date**: 2025-07-25  
**Content Quality**: ✅ PASSED (1,575+ characters, comprehensive technical content)

## Overview

Each day, Alternative.me analyzes emotions and sentiments from different sources and crunches them into one simple number: The Fear & Greed Index for Bitcoin and other large cryptocurrencies.

The index ranges from 0 to 100:
- **0**: Extreme Fear
- **100**: Extreme Greed

## Current Index Values

- **Now**: Greed (70)
- **Yesterday**: Greed (71) 
- **Last week**: Greed (73)
- **Last month**: Greed (66)

## Market Psychology & Strategy

### Why Measure Fear and Greed?

The crypto market behaviour is very emotional. People tend to get greedy when the market is rising which results in FOMO (Fear of missing out). Also, people often sell their coins in irrational reaction of seeing red numbers. With the Fear and Greed Index, investors can save themselves from emotional overreactions.

### Two Key Trading Assumptions:

1. **Extreme fear** can be a sign that investors are too worried. That could be a **buying opportunity**.
2. When Investors are getting **too greedy**, that means the market is **due for a correction**.

## Data Sources & Methodology

The index is calculated from 6 different factors, each weighted differently:

### 1. Volatility (25%)
- **Method**: Measures current volatility and max drawdowns of bitcoin
- **Comparison**: Against 30-day and 90-day average values
- **Logic**: Unusual rise in volatility indicates a fearful market

### 2. Market Momentum/Volume (25%)
- **Method**: Analyzes current volume and market momentum
- **Comparison**: Against 30/90 day average values
- **Logic**: High buying volumes in positive market = overly greedy/bullish behavior

### 3. Social Media (15%)
- **Platform**: Twitter analysis (Reddit analysis in development)
- **Method**: Gathers and counts posts on various hashtags for each coin
- **Metrics**: Interaction speed and quantity in specific time frames
- **Logic**: Unusual high interaction rate = grown public interest = greedy market behavior

### 4. Surveys (15%) - Currently Paused
- **Platform**: strawpoll.com partnership
- **Method**: Weekly crypto polls asking market sentiment
- **Sample Size**: Typically 2,000-3,000 votes per poll
- **Status**: Paused but historically useful for sentiment validation

### 5. Dominance (10%)
- **Method**: Bitcoin market cap share analysis
- **Logic For Bitcoin**: Rise in Bitcoin dominance = fear of speculative alt-coins (Bitcoin as safe haven)
- **Logic For Alt-coins**: Decreasing Bitcoin dominance = increased greed (more risky investments)

### 6. Trends (10%)
- **Source**: Google Trends data for Bitcoin-related search queries
- **Metrics**: Search volume changes and related popular searches
- **Example**: +1,550% rise in "bitcoin price manipulation" searches indicates market fear

## API Integration

### Base API Information
- **API Base URL**: https://api.alternative.me/
- **Endpoint**: /fng/
- **Method**: GET
- **Rate Limits**: Not specified (free tier)
- **Attribution Required**: Yes, for commercial use

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 1 | Number of results (0 = all data) |
| `format` | string | json | Response format: 'json' or 'csv' |
| `date_format` | string | unixtime | Date format: 'us', 'cn', 'kr', 'world' |

### Example API Calls

```bash
# Latest value
GET https://api.alternative.me/fng/

# Last 10 values
GET https://api.alternative.me/fng/?limit=10

# CSV format with US date format
GET https://api.alternative.me/fng/?limit=10&format=csv&date_format=us
```

### Response Schema

```json
{
  "name": "Fear and Greed Index",
  "data": [
    {
      "value": "40",
      "value_classification": "Fear",
      "timestamp": "1551157200",
      "time_until_update": "68499"  // Only in latest value
    }
  ],
  "metadata": {
    "error": null
  }
}
```

### Value Classifications
- **0-25**: Extreme Fear
- **26-45**: Fear  
- **46-54**: Neutral
- **55-75**: Greed
- **76-100**: Extreme Greed

## Implementation Considerations for Quant Trading

### Market Regime Detection
- **Contrarian Signals**: Extreme fear (0-25) = potential buy zones, extreme greed (75-100) = correction signals
- **Genetic Algorithm Input**: Use as environmental pressure for strategy adaptation
- **Real-time Integration**: API polling for dynamic strategy selection

### Data Integration Pattern
```python
# Example integration pattern
import requests

def get_fear_greed_index(limit=1):
    """Fetch Fear & Greed Index data"""
    url = f"https://api.alternative.me/fng/?limit={limit}"
    response = requests.get(url)
    return response.json()

# Multi-factor analysis integration
def analyze_market_regime(fng_data, volume_data, volatility_data):
    """Combine FnG with other market indicators"""
    fng_value = int(fng_data['data'][0]['value'])
    
    if fng_value <= 25:
        return "extreme_fear_buy_opportunity"
    elif fng_value >= 75:
        return "extreme_greed_correction_risk"
    else:
        return "neutral_trend_following"
```

### Historical Data Access
- API provides historical data back to index inception
- Use `limit=0` to get all historical values
- Timestamp format: Unix timestamp (seconds since epoch)

## Widget & Embedding Options

### Self-Updating Image Widget
```html
<img src="https://alternative.me/crypto/fear-and-greed-index.png" 
     alt="Latest Crypto Fear & Greed Index" />
```

### iOS Widget Available
- Requires Scriptable app (free)
- Provides home screen widget functionality
- Real-time index updates

## Attribution Requirements

- **Commercial Use**: Allowed with proper attribution
- **Attribution Placement**: Must be prominent and next to data display
- **Impersonation**: Not allowed to create confusing competing services
- **Source Reference**: Must acknowledge Alternative.me as data source

## Technical Notes

- **Update Frequency**: Daily updates (specific time not documented)
- **Primary Focus**: Bitcoin-centric but expanding to other large cryptocurrencies
- **Data Retention**: Full historical data available via API
- **Reliability**: Established service with consistent methodology since launch

## Use Cases for Algorithmic Trading

1. **Sentiment-Based Strategy Switching**: Adjust algorithm aggressiveness based on market sentiment
2. **Contrarian Signal Generation**: Generate buy/sell signals from extreme fear/greed readings
3. **Risk Management**: Reduce position sizes during extreme greed periods
4. **Portfolio Rebalancing**: Use sentiment as factor in allocation decisions
5. **Market Timing**: Combine with technical indicators for entry/exit timing