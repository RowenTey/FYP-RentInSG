# FYP

## Connecting to EC2 instance

```bash
ssh -i "secrets/AWS.pem" ec2-user@ec2-13-250-116-9.ap-southeast-1.compute.amazonaws.com
```

## Transferring files

```bash
sftp -i "secrets/AWS.pem" ec2-user@ec2-13-250-116-9.ap-southeast-1.compute.amazonaws.com
```

## Script Timings

| Name          | UTC Time | SGT Time | Remarks            |
| ------------- | -------- | -------- | ------------------ |
| 99.co scraper | 2am      | 10am     | Takes ~4hrs to run |
| s3_uploader   | 6am      | 2pm      |                    |

## Changing Permissions

```bash
chmod 777 /path/to/directory
```

## List processes

```bash
ps -aux
```

## Kill process

```bash
kill <process-id>
```
