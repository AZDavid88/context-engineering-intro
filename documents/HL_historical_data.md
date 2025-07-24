# Historical data

The examples below use the AWS CLI (see [https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)) and LZ4 ([https://github.com/lz4/lz4](https://github.com/lz4/lz4) or install from your package manager).

Note that the requester of the data must pay for transfer costs.

### Asset data

Historical data is uploaded to the bucket `hyperliquid-archive`  approximately once a month. L2 book snapshots are available in market\_data and asset contexts are available in asset\_ctxs. No other historical data sets are provided via S3 (e.g. candles or spot asset data). You can use the API to record additional historical data sets yourself.&#x20;

Format: `s3://hyperliquid-archive/market_data/[date]/[hour]/[datatype]/[coin].lz4` or `s3://hyperliquid-archive/asset_ctxs/[date].csv.lz4`

```
aws s3 cp s3://hyperliquid-archive/market_data/20230916/9/l2Book/SOL.lz4 /tmp/SOL.lz4 --request-payer requester
unlz4 --rm /tmp/SOL.lz4
head /tmp/SOL
```

### Trade data

`s3://hl-mainnet-node-data/node_fills` has fills which are streamed via `--write-fills` from a non-validating node. Older data is in a different format at `s3://hl-mainnet-node-data/node_trades` . `node_fills` matches the API format, while `node_trades` does not.

### Historical node data

`s3://hl-mainnet-node-data/explorer_blocks`and `s3://hl-mainnet-node-data/replica_cmds` contain historical explorer blocks and L1 transactions. &#x20;
