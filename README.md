# FYP

## Connecting to EC2 instance

```bash
ssh -i "AWS.pem" ec2-user@ec2-13-250-116-9.ap-southeast-1.compute.amazonaws.com
```

## Script Timings

| Name          | UTC Time | SGT Time | Remarks |
| ------------- | -------- | -------- | ------- |
| 99.co scraper |          |          |         |
|               |          |          |         |

## Architecture

```mermaid
erDiagram
  entity "Property Listing" {
    +listing_id [PK]
    property_name
    district
    price
    bedroom
    bathroom
    dimensions
    address
    latitude
    longitude
    price_sqft
    floor_level
    furnishing
    facing
    built_year
    tenure
    property_type
    url
  }
```
