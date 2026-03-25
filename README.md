<!-- BlackRoad SEO Enhanced -->

# ulackroad image optimizer

> Part of **[BlackRoad OS](https://blackroad.io)** — Sovereign Computing for Everyone

[![BlackRoad OS](https://img.shields.io/badge/BlackRoad-OS-ff1d6c?style=for-the-badge)](https://blackroad.io)
[![BlackRoad Media](https://img.shields.io/badge/Org-BlackRoad-Media-2979ff?style=for-the-badge)](https://github.com/BlackRoad-Media)
[![License](https://img.shields.io/badge/License-Proprietary-f5a623?style=for-the-badge)](LICENSE)

**ulackroad image optimizer** is part of the **BlackRoad OS** ecosystem — a sovereign, distributed operating system built on edge computing, local AI, and mesh networking by **BlackRoad OS, Inc.**

## About BlackRoad OS

BlackRoad OS is a sovereign computing platform that runs AI locally on your own hardware. No cloud dependencies. No API keys. No surveillance. Built by [BlackRoad OS, Inc.](https://github.com/BlackRoad-OS-Inc), a Delaware C-Corp founded in 2025.

### Key Features
- **Local AI** — Run LLMs on Raspberry Pi, Hailo-8, and commodity hardware
- **Mesh Networking** — WireGuard VPN, NATS pub/sub, peer-to-peer communication
- **Edge Computing** — 52 TOPS of AI acceleration across a Pi fleet
- **Self-Hosted Everything** — Git, DNS, storage, CI/CD, chat — all sovereign
- **Zero Cloud Dependencies** — Your data stays on your hardware

### The BlackRoad Ecosystem
| Organization | Focus |
|---|---|
| [BlackRoad OS](https://github.com/BlackRoad-OS) | Core platform and applications |
| [BlackRoad OS, Inc.](https://github.com/BlackRoad-OS-Inc) | Corporate and enterprise |
| [BlackRoad AI](https://github.com/BlackRoad-AI) | Artificial intelligence and ML |
| [BlackRoad Hardware](https://github.com/BlackRoad-Hardware) | Edge hardware and IoT |
| [BlackRoad Security](https://github.com/BlackRoad-Security) | Cybersecurity and auditing |
| [BlackRoad Quantum](https://github.com/BlackRoad-Quantum) | Quantum computing research |
| [BlackRoad Agents](https://github.com/BlackRoad-Agents) | Autonomous AI agents |
| [BlackRoad Network](https://github.com/BlackRoad-Network) | Mesh and distributed networking |
| [BlackRoad Education](https://github.com/BlackRoad-Education) | Learning and tutoring platforms |
| [BlackRoad Labs](https://github.com/BlackRoad-Labs) | Research and experiments |
| [BlackRoad Cloud](https://github.com/BlackRoad-Cloud) | Self-hosted cloud infrastructure |
| [BlackRoad Forge](https://github.com/BlackRoad-Forge) | Developer tools and utilities |

### Links
- **Website**: [blackroad.io](https://blackroad.io)
- **Documentation**: [docs.blackroad.io](https://docs.blackroad.io)
- **Chat**: [chat.blackroad.io](https://chat.blackroad.io)
- **Search**: [search.blackroad.io](https://search.blackroad.io)

---


> Image optimization, resizing, and CDN pipeline

Part of the [BlackRoad OS](https://blackroad.io) ecosystem — [BlackRoad-Media](https://github.com/BlackRoad-Media)

---

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
