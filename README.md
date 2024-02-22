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
|               |          |          |                    |

## Permissions on Linux

1. The first character represents the file type:

   - d indicates a directory.
   - \- indicates a regular file.

2. The next three characters represent the permissions for the owner of the file/directory:

   - The first character (r) indicates read permission.
   - The second character (w) indicates write permission.
   - The third character (x) indicates execute permission.

3. The next three characters represent the permissions for the group associated with the file/directory:

   - The first character (r) indicates read permission.
   - The second character (w) indicates write permission.
   - The third character (x) indicates execute permission.

4. The last three characters represent the permissions for others (users not in the owner group or file owner):

   - The first character (r) indicates read permission.
   - The second character (w) indicates write permission.
   - The third character (x) indicates execute permission.

NOTE - Octal format: 4 (Read) + 2 (Write) + 1 (Execute)

```bash
chmod 777 /path/to/directory
```
