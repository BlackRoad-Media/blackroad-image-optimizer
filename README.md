# BlackRoad Image Optimizer

> Image optimization, resizing, and CDN pipeline — part of the BlackRoad Media suite.

[![CI](https://github.com/BlackRoad-Media/blackroad-image-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/BlackRoad-Media/blackroad-image-optimizer/actions/workflows/ci.yml)

## Features

- **Multi-format output**: JPEG, WebP, AVIF, PNG
- **Smart resizing**: Area-ratio based size calculation (no external dependencies)
- **CDN integration**: Automatic CDN URL generation and cache management
- **Responsive sets**: Auto-generate mobile/tablet/desktop/retina variants
- **srcset generation**: HTML-ready srcset attribute strings
- **Batch processing**: Optimize multiple images at once
- **Savings estimation**: Pre-flight compression estimation
- **SQLite persistence**: Jobs, variants, and CDN entries stored

## Quick Start

```bash
pip install -r requirements.txt
python image_optimizer.py
```

## Usage

```python
from image_optimizer import create_optimizer

opt = create_optimizer()

# Create responsive image set
job = opt.create_responsive_set("https://example.com/hero.jpg")
result = opt.process(job.id)

# Get HTML srcset
srcset = opt.generate_srcset(job.id)
# → "https://cdn.../mobile.webp 480w, https://cdn.../desktop.webp 1200w, ..."

# Batch process
results = opt.optimize_batch(["https://example.com/img1.jpg", "https://example.com/img2.jpg"])

# Estimate savings
estimate = opt.estimate_savings("https://example.com/large.jpg", "avif", 80)
print(f"Estimated savings: {estimate['estimated_savings_percent']}%")
```

## Responsive Variants

| Name | Width | Format | Quality |
|------|-------|--------|---------|
| mobile | 480px | WebP | 80 |
| tablet | 768px | WebP | 82 |
| desktop | 1200px | WebP | 85 |
| retina | 2400px | WebP | 85 |
| mobile-jpeg | 480px | JPEG | 75 |
| desktop-avif | 1200px | AVIF | 80 |

## Testing

```bash
pytest tests/ -v --cov=image_optimizer
```

## License

Proprietary — © BlackRoad OS, Inc.
