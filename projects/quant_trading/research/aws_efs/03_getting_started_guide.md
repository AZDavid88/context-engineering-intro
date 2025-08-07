# AWS EFS Getting Started Guide - Phase 4 Implementation

**Source**: https://docs.aws.amazon.com/efs/latest/ug/getting-started.html  
**Extraction Date**: 2025-08-06  
**Project Context**: Phase 4 - Practical EFS Setup for Ray Worker Integration

## Quick Start Overview

For Phase 4 implementation, follow these steps to establish AWS EFS as interim cloud storage for Ray workers:

1. **Prerequisites Validation** - Ensure AWS setup and VPC configuration
2. **EFS File System Creation** - Create shared file system with recommended settings
3. **EC2 Integration** - Launch Ray worker instances with automatic EFS mounting
4. **Data Transfer Setup** - Configure AWS DataSync for data migration (if needed)
5. **Resource Management** - Proper cleanup and cost control procedures

## Prerequisites Checklist

**AWS Account Setup:**
- ✅ AWS account with administrative access
- ✅ Familiar with launching EC2 instances
- ✅ EC2 key pair configured
- ✅ Security group configured for NFS access

**Network Configuration:**
- ✅ Default VPC available in target AWS Region
- ✅ Amazon VPC, EC2, and EFS resources in same AWS Region
- ✅ Default inbound access rule unchanged for default security group
- ✅ **Critical**: Ray workers and EFS must be in same region

**Phase 4 Specific Requirements:**
- ✅ Docker environment prepared for Ray worker containers
- ✅ Ray cluster configuration compatible with shared file system
- ✅ Understanding of Phase 1-3 DataStorageInterface for seamless integration

## EFS File System Creation

**Recommended Method: EC2 Launch Wizard Integration**

The fastest approach is to create EFS file system during EC2 instance launch, which automatically:
1. Creates EC2 instance running Linux (required - no Windows support)
2. Creates shared EFS file system with recommended settings
3. Automatically mounts EFS to EC2 instance
4. Launches instance with EFS readily available

**Alternative: EFS Console Creation**
- Amazon EFS console: Create file systems with recommended or custom settings
- AWS CLI/API: Programmatic file system creation for automation
- Custom configuration options for advanced Phase 4 requirements

### Phase 4 Ray Worker Integration Steps

**Step 1: Create EFS File System**
```bash
# AWS CLI approach for automation
aws efs create-file-system \
    --performance-mode generalPurpose \
    --throughput-mode elastic \
    --encrypted \
    --tags Key=Project,Value=QuantTradingPhase4 \
         Key=Purpose,Value=RayWorkerSharedStorage
```

**Step 2: Create Mount Targets**
```bash
# Create mount target in each AZ where Ray workers operate
aws efs create-mount-target \
    --file-system-id fs-xxxxxxxxx \
    --subnet-id subnet-xxxxxxxxx \
    --security-groups sg-xxxxxxxxx
```

**Step 3: Docker Compose Configuration**
```yaml
# docker-compose.yml for Ray workers with EFS
version: '3.8'
services:
  ray-worker:
    image: rayproject/ray:latest
    volumes:
      - type: nfs
        source: fs-xxxxxxxxx.efs.us-east-1.amazonaws.com
        target: /shared_data
        nfs_opts: "nfsvers=4.1,rsize=1048576,wsize=1048576"
    environment:
      - RAY_SHARED_DATA_PATH=/shared_data
```

## Data Transfer with AWS DataSync

**When to Use DataSync:**
- Migrating existing Phase 1-3 data to cloud storage
- Initial population of EFS with historical trading data
- Ongoing synchronization between on-premises and cloud data

### Prerequisites for DataSync Integration

**Source System Requirements:**
- NFS version 3, 4, or 4.1 accessible source file system
- Examples: On-premises data center, self-managed cloud systems, existing EFS
- Network connectivity to AWS (internet or AWS Direct Connect)

**DataSync Setup Requirements:**
- AWS DataSync agent deployed in source environment
- Proper IAM permissions for cross-service access
- Source and destination location configuration

### DataSync Implementation Steps

**Step 1: Agent Deployment**
```bash
# Download and deploy DataSync agent (on-premises or EC2)
# Agent connects source location to AWS DataSync service
```

**Step 2: Location Configuration**
```bash
# Create source location (existing file system)
aws datasync create-location-nfs \
    --server-hostname source.example.com \
    --subdirectory /path/to/trading/data

# Create destination location (EFS file system)  
aws datasync create-location-efs \
    --efs-file-system-arn arn:aws:elasticfilesystem:region:account:file-system/fs-xxxxxxxxx
```

**Step 3: Task Configuration and Execution**
```bash
# Create DataSync task for data transfer
aws datasync create-task \
    --source-location-arn arn:aws:datasync:region:account:location/loc-source \
    --destination-location-arn arn:aws:datasync:region:account:location/loc-dest \
    --options VerifyMode=POINT_IN_TIME_CONSISTENT

# Execute data transfer task
aws datasync start-task-execution --task-arn arn:aws:datasync:region:account:task/task-xxxxxxxxx
```

## Resource Management and Cleanup

**Phase 4 Development Lifecycle Management:**

### During Development
```bash
# Monitor EFS usage and costs
aws efs describe-file-systems --file-system-id fs-xxxxxxxxx
aws cloudwatch get-metric-statistics --namespace AWS/EFS --metric-name StorageBytes
```

### Cleanup Procedures
```bash
# 1. Unmount EFS from all Ray workers
sudo umount /shared_data

# 2. Delete EFS file system (WARNING: All data lost)
aws efs delete-file-system --file-system-id fs-xxxxxxxxx

# 3. Terminate Ray worker EC2 instances
aws ec2 terminate-instances --instance-ids i-xxxxxxxxx

# 4. Clean up security groups (if created specifically for Phase 4)
aws ec2 delete-security-group --group-id sg-xxxxxxxxx
```

### Cost Control Best Practices

**Monitoring and Budgets:**
- Set up CloudWatch billing alarms for EFS costs
- Use AWS Cost Explorer to track EFS usage patterns
- Implement lifecycle policies to move data to IA (Infrequent Access) storage

**Storage Optimization:**
```bash
# Configure lifecycle policy for cost optimization
aws efs put-lifecycle-configuration \
    --file-system-id fs-xxxxxxxxx \
    --lifecycle-policies \
    TransitionToIA=AFTER_30_DAYS,TransitionToPrimaryStorageClass=AFTER_1_ACCESS
```

## Phase 4 Integration Patterns

**DataStorageInterface Compatibility:**
```python
# Phase 4 EFS integration with existing DataStorageInterface
class EFSDataStorage(DataStorageInterface):
    def __init__(self, efs_mount_path="/shared_data"):
        self.mount_path = efs_mount_path
        # EFS appears as standard filesystem to Ray workers
        
    async def store_ohlcv_bars(self, bars):
        # Write to EFS-mounted path using standard file operations
        file_path = f"{self.mount_path}/ohlcv/{symbol}.parquet"
        # Standard Parquet write operations work seamlessly
```

**Ray Worker Configuration:**
```python
# Ray workers automatically access shared EFS storage
@ray.remote  
class GeneticAlgorithmWorker:
    def __init__(self):
        self.shared_data_path = "/shared_data"  # EFS mount point
        
    def process_generation(self, population_data):
        # Read shared data from EFS
        historical_data = pd.read_parquet(f"{self.shared_data_path}/historical/")
        # Process genetic algorithm with shared data access
        return processed_results
```

## Security Configuration

**EFS Security Group Rules:**
```bash
# Allow NFS traffic from Ray worker security group
aws ec2 authorize-security-group-ingress \
    --group-id sg-efs-mount-target \
    --protocol tcp \
    --port 2049 \
    --source-group sg-ray-workers
```

**Encryption Configuration:**
- **At Rest**: Enable during EFS file system creation
- **In Transit**: Configure during mount operation with TLS
- **IAM Integration**: Control API access to EFS resources

## Troubleshooting Common Issues

**Mount Failures:**
- Verify security group allows NFS traffic (port 2049)
- Check VPC DNS resolution settings
- Ensure NFS client installed on Ray worker containers

**Performance Issues:**
- Monitor CloudWatch metrics for throughput bottlenecks
- Optimize mount options for Ray worker access patterns
- Consider Regional vs One Zone file system performance characteristics

**Cost Overruns:**
- Implement lifecycle policies for infrequently accessed data
- Monitor storage utilization patterns
- Set up billing alerts for unexpected usage spikes

## Phase 4 Success Criteria

- ✅ **EFS File System**: Created with Regional redundancy and encryption
- ✅ **Mount Targets**: Available in all AZs where Ray workers operate
- ✅ **Ray Worker Integration**: Seamless shared data access across workers
- ✅ **Performance Validation**: General Purpose mode meets trading workload requirements
- ✅ **Cost Control**: Lifecycle policies and monitoring in place
- ✅ **Security Compliance**: Encryption and access controls configured
- ✅ **DataSync Ready**: Optional data migration capability validated

**Next Steps**: Proceed to [Performance Specifications](./04_performance_specifications.md) for optimization guidance, then [Security Considerations](./05_security_considerations.md) for compliance requirements.