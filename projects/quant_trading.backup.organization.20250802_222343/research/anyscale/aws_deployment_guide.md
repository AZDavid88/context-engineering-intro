# Anyscale AWS Deployment Guide

## AWS Deployment Overview
Anyscale integrates with AWS to manage Ray clusters directly in your AWS account, providing seamless access to EC2 instances, S3 storage, and other AWS services while maintaining security and cost control.

## Deployment Methods

### Automated Setup (`anyscale cloud setup`)
- **Use Case**: Rapid deployment with minimal configuration
- **Network Architecture**: Direct networking with public subnets
- **Resource Management**: Anyscale automatically creates and configures AWS resources
- **Access Model**: Public IP addresses without additional networking infrastructure
- **Ideal For**: Development environments, proof of concepts, getting started

### Custom Registration (`anyscale cloud register`)
- **Use Case**: Enterprise environments with advanced requirements
- **Network Architecture**: Customer-defined networking with private subnets
- **Resource Management**: Customer creates and manages all AWS resources
- **Access Model**: Public or private IP addresses with custom networking
- **Ideal For**: Production environments, compliance requirements, custom security

## AWS Resource Requirements

### Core Infrastructure Components

#### Virtual Private Cloud (VPC)
- **CIDR Range**: /19 or larger recommended for adequate IP space
- **Internet Connectivity**: Must have internet egress capability
- **Gateway Endpoint**: S3 VPC endpoint recommended for cost reduction and performance
- **DNS Support**: Enable DNS hostnames and DNS resolution

#### Subnets
- **CIDR Range**: /22 or larger per subnet
- **Minimum Count**: 2 subnets required for high availability
- **Availability Zones**: No two public subnets in same AZ
- **Routing**: Public subnets with internet gateway, private with NAT gateway

#### Security Groups
**Inbound Rules:**
- Port 443: HTTPS access for job submission, Grafana, workspaces, VS Code
- Self-referencing: All traffic from same security group for intra-cluster communication

**Outbound Rules:**
- All traffic: Required for Anyscale control plane communication
- Self-referencing: Required for Ray cluster communication and EFA networking

### IAM Configuration

#### Cross-Account Access Role (`anyscale-iam-role-id`)
**Trust Relationship:**
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "sts:AssumeRole",
    "Principal": {"AWS": "525325868955"},
    "Condition": {} // Populated with External ID after registration
  }]
}
```

**Required Permissions:**
- **Instance Management**: EC2 instance lifecycle, spot instances, placement groups
- **Resource Management**: Volumes, addresses, IAM instance profiles
- **EFS Access**: Mount target discovery
- **Service Linked Roles**: Create EC2 spot service role if needed

**Additional for Services:**
- **CloudFormation**: Stack management for load balancers
- **ELB Management**: Load balancer lifecycle and configuration
- **ACM Integration**: SSL certificate management
- **IAM Policies**: Service-linked role management

#### Cluster Node Role (`instance-iam-role-id`)
**Purpose**: Default role attached to Ray cluster instances
**Trust Policy**: Allow EC2 service to assume role
**Minimum Permissions**: S3 bucket read/write access
**Instance Profile**: Must exist with same name as role

**S3 Access Example:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListObjectsInBucket",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": ["arn:aws:s3:::bucket-name"]
    },
    {
      "Sid": "AllObjectActions",
      "Effect": "Allow",
      "Action": "s3:*Object",
      "Resource": ["arn:aws:s3:::bucket-name/*"]
    }
  ]
}
```

### Storage Configuration

#### S3 Bucket
**Naming Convention**: `anyscale-production-data-{cloud_id}` (customizable)
**Data Organization:**
- `{organization_id}/`: Anyscale managed data
- `{organization_id}/{cloud_id}/logs`: Cluster and application logs
- `/logs`: Legacy log location (being migrated)

**Bucket Policy Example:**
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "allow-role-access",
    "Effect": "Allow",
    "Principal": {
      "AWS": [
        "arn:aws:iam::<account_id>:role/<anyscale-iam-role>",
        "arn:aws:iam::<account_id>:role/<instance-iam-role>"
      ]
    },
    "Action": [
      "s3:PutObject", "s3:DeleteObject", "s3:GetObject",
      "s3:ListBucket", "s3:ListBucketMultipartUploads",
      "s3:ListMultipartUploadParts", "s3:AbortMultipartUpload",
      "s3:GetBucketLocation"
    ],
    "Resource": [
      "arn:aws:s3:::<bucket-name>/*",
      "arn:aws:s3:::<bucket-name>"
    ]
  }]
}
```

**CORS Configuration** (for UI access):
```json
[{
  "AllowedHeaders": ["*"],
  "AllowedMethods": ["GET", "PUT", "POST", "HEAD", "DELETE"],
  "AllowedOrigins": ["https://*.anyscale.com"],
  "ExposeHeaders": []
}]
```

#### Elastic File System (EFS)
- **Purpose**: Shared storage for Anyscale workspaces
- **Mount Targets**: Configure in all subnets with security group access
- **Performance**: General Purpose or Provisioned Throughput based on needs
- **Security**: NFS access through security group rules

#### MemoryDB (Optional)
- **Purpose**: Enable head node fault tolerance for services
- **Instance Type**: `db.t4g.small` minimum, larger for production
- **Configuration**: Same VPC/subnets, associated security group
- **High Availability**: Minimum 1 replica per shard (2 nodes total)
- **Security**: TLS enabled, maxmemory-policy set to allkeys-lru

## Deployment Process

### Prerequisites
1. **AWS CLI**: Configured with appropriate credentials
2. **Anyscale CLI**: Installed and authenticated (`anyscale login`)
3. **Permissions**: IAM permissions for resource creation
4. **Quotas**: Verify VPC and internet gateway quotas

### Automated Deployment
```bash
anyscale cloud setup \
  --name example_cloud_name \
  --provider aws \
  --region us-west-2 \
  --enable-head-node-fault-tolerance
```

### Custom Deployment
1. **Create Resources**: Use Terraform module or manual creation
2. **Register Cloud**: Use `anyscale cloud register` with resource IDs
3. **Verify Configuration**: Run `anyscale cloud verify`
4. **Update Permissions**: Apply cloud-specific resource tagging

### Verification Process
```bash
anyscale cloud verify --name my-cloud-deployment --functional-verify workspace,service
```

**Verification Steps:**
- VPC and subnet validation
- IAM role and policy verification
- Security group rule checking
- S3 bucket access testing
- EFS mount target validation
- Optional functional testing with workspace/service launch

## Security Considerations

### Network Security
- **Principle of Least Privilege**: Minimal required access
- **Security Group Rules**: Specific port and protocol restrictions
- **Private Networking**: Use private subnets for enhanced security
- **VPC Endpoints**: Reduce internet traffic with AWS service endpoints

### Access Control
- **IAM Roles**: Separate roles for control plane and data plane
- **External ID**: Unique identifier for cross-account trust
- **Resource Tagging**: Cloud-specific tags for resource isolation
- **Audit Logging**: CloudTrail integration for API call tracking

### Data Protection
- **Encryption**: S3 bucket encryption (SSE-S3 or SSE-KMS)
- **Data Lifecycle**: Configure S3 lifecycle policies for log retention
- **Backup Strategy**: Consider S3 versioning for critical data
- **Access Monitoring**: CloudWatch metrics and S3 access logging

## Cost Optimization

### Resource Optimization
- **Spot Instances**: Up to 90% cost savings for fault-tolerant workloads
- **Right-sizing**: Match instance types to workload requirements
- **Auto-scaling**: Dynamic resource allocation based on demand
- **Reserved Instances**: Commitment-based savings for predictable workloads

### Storage Optimization
- **S3 Storage Classes**: Intelligent tiering for cost-effective storage
- **Lifecycle Policies**: Automatic transition to cheaper storage classes
- **Data Compression**: Reduce storage costs through compression
- **Log Retention**: Configure appropriate log retention periods

### Monitoring and Alerts
- **Cost Budgets**: AWS Budget integration with Anyscale usage
- **Resource Tagging**: Track costs by project, team, or environment
- **Usage Analytics**: Anyscale usage dashboard for optimization insights
- **Automated Cleanup**: Terminate idle resources automatically

This comprehensive AWS deployment guide provides the foundation for secure, scalable, and cost-effective Ray cluster management in enterprise AWS environments.