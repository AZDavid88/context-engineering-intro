# AWS EFS Architecture & Implementation - Phase 4 Technical Details

**Source**: https://docs.aws.amazon.com/efs/latest/ug/how-it-works.html  
**Extraction Date**: 2025-08-06  
**Project Context**: Phase 4 - EFS Architecture for Distributed Ray Worker Access

## Core Architecture Overview

Amazon EFS provides a simple, serverless, set-and-forget elastic file system that integrates seamlessly with Amazon EC2 instances through Network File System versions 4.0 and 4.1 (NFSv4) protocol.

### Mount Target Architecture

**Mount Target Fundamentals:**
- **Mount Target**: Provides an IP address for an NFSv4 endpoint to mount an EFS file system
- **DNS Resolution**: Mount using DNS name which resolves to IP address of EFS mount target
- **Availability Zone Placement**: Create one mount target in each Availability Zone
- **High Availability**: Mount targets are designed to be highly available with redundant resources

**Critical Architecture Constraint:**
> An EFS file system can have mount targets in only one VPC at a time.

### Regional vs One Zone File Systems

#### Regional EFS File Systems (Recommended for Phase 4)

**Architecture Benefits:**
- Multiple EC2 instances access EFS file system across multiple Availability Zones
- Data stored redundantly across geographically separated AZs within same AWS Region
- Continuous availability even when one or more AZs are unavailable
- Ideal for distributed Ray worker deployments

**Ray Worker Integration Pattern:**
```
Region: us-east-1
├── AZ-1a: Ray Workers + EFS Mount Target
├── AZ-1b: Ray Workers + EFS Mount Target  
├── AZ-1c: Ray Workers + EFS Mount Target
└── Shared EFS File System: Cross-AZ Data Access
```

#### One Zone File Systems (Cost-Optimized Alternative)

**Architecture Characteristics:**
- Store data within single Availability Zone
- Lower cost compared to Regional file systems
- Continuous availability within single AZ
- Risk: Data loss if entire AZ becomes unavailable

### Integration with AWS Services

#### Amazon EC2 Integration

**Multi-Instance Access Pattern:**
```
Multiple EC2 Instances (Ray Workers)
         ↓ NFSv4.1 Protocol
    EFS File System
         ↓ Cross-AZ Replication
[AZ-1a] [AZ-1b] [AZ-1c] Mount Targets
```

**Technical Implementation Details:**
- Concurrent access from multiple NFS clients
- Applications scale beyond single connection to access file system
- Many Ray workers can access and share common data source
- POSIX-compliant file system behavior after mounting

#### VPC Network Integration

**Mount Target Configuration:**
- One mount target per Availability Zone in VPC
- If multiple subnets in AZ, create mount target in one subnet
- All EC2 instances in that AZ share the mount target
- Static IP addresses and DNS for mount targets (redundant components)

### On-Premises Integration (Advanced Use Case)

**Hybrid Cloud Capabilities:**
- Mount EFS file systems on on-premises data center servers
- Requires AWS Direct Connect or AWS VPN connection to Amazon VPC
- Use cases: Dataset migration to EFS, cloud bursting, on-premises backup

### Performance Architecture

**Throughput and IOPS Characteristics:**
- EFS file systems grow to petabyte scale
- Drive high levels of throughput
- Allow massively parallel access from compute instances
- Recommended: General Purpose performance mode + Elastic throughput mode

**For Phase 4 Ray Workers:**
- **General Purpose Mode**: Ideal for latency-sensitive trading applications
- **Elastic Throughput**: Automatically scales to meet Ray worker activity demands

### Security Architecture

**Multi-Layer Security Model:**
1. **Network Security**: VPC security groups control mount target access
2. **IAM Policies**: Control API-level access to EFS resources
3. **NFS Permissions**: POSIX permissions for file-level access control
4. **Encryption**: Both in-transit and at-rest encryption available

### Phase 4 Implementation Architecture

**Current Phase 1-3 to Phase 4 Transition:**
```
Phase 1-3 (Local):
Ray Workers → Local DuckDB → Parquet Files

Phase 4 Interim (EFS):
Ray Workers → EFS Mount → Shared Data Access → Proven Cloud System

Phase 4 Target (Neon):
Ray Workers → Neon Database → TimescaleDB → Enhanced Performance
```

**EFS Integration Benefits for Phase 4:**
1. **Immediate Cloud Deployment**: No database setup complexity
2. **Proven NFS Compatibility**: Ray workers can mount as standard filesystem  
3. **Cross-AZ Resilience**: Ray workers operate across multiple AZs safely
4. **Scalable Storage**: Automatic scaling eliminates capacity planning
5. **Cost-Effective Interim**: Pay-per-use model during development

### Technical Requirements for Ray Workers

**NFS Client Requirements:**
- Current generation Linux NFSv4.1 client recommended
- Compatible with Amazon Linux, Amazon Linux 2, Red Hat, Ubuntu AMIs  
- EFS mount helper installation for optimal performance
- For some AMIs, NFS client installation required

**Mount Configuration for Docker Containers:**
```yaml
# Optimized EFS mount for Ray workers
volumes:
  - type: nfs
    source: ${EFS_DNS_NAME}
    target: /shared_data
    nfs_opts: "nfsvers=4.1,rsize=1048576,wsize=1048576,hard,intr,timeo=600"
```

**Performance Optimization Settings:**
- **rsize/wsize**: 1MB read/write buffer sizes for throughput
- **hard mount**: Ensures operations complete even during network issues
- **intr**: Allows interruption of hung operations
- **timeo=600**: 60-second timeout for resilient operations

### Monitoring and Observability

**CloudWatch Integration:**
- File system metrics available in Amazon CloudWatch
- Mount attempt success/failure monitoring
- Performance metrics for throughput optimization
- Custom alarms for operational monitoring

### Phase 4 Implementation Checklist

- ✅ **Regional File System**: Multi-AZ redundancy for Ray worker resilience
- ✅ **Mount Targets**: One per AZ where Ray workers operate
- ✅ **VPC Security Groups**: Proper NFS port access (2049) configuration  
- ✅ **DNS-based Mounting**: Use EFS DNS name for automatic failover
- ✅ **Performance Mode**: General Purpose + Elastic for trading workloads
- ✅ **NFS Client**: Latest NFSv4.1 client on Ray worker containers
- ✅ **Mount Options**: Optimized settings for distributed workloads

**Next Steps**: Proceed to [Getting Started Guide](./03_getting_started_guide.md) for practical implementation procedures.