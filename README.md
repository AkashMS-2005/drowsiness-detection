# \# 🚗 Real-Time Driver Drowsiness Detection System

# 

# !\[Python](https://img.shields.io/badge/Python-3.13-blue)

# !\[TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)

# !\[OpenCV](https://img.shields.io/badge/OpenCV-4.13-green)

# !\[Accuracy](https://img.shields.io/badge/Accuracy-96.7%25-brightgreen)

# 

# \## 📌 Overview

# A deep learning based real-time driver drowsiness detection system

# that uses CNN and OpenCV to detect drowsy drivers and alert them

# before accidents occur.

# 

# \## 🎯 Features

# \- ✅ Real-time webcam detection

# \- ✅ CNN model with 96.7% accuracy

# \- ✅ Trained on 48,000 eye images

# \- ✅ Instant audio alert system

# \- ✅ Drowsiness level progress bar

# \- ✅ Works with glasses

# \- ✅ Face region based eye extraction

# 

# \## 🛠️ Tech Stack

# | Technology | Purpose |

# |------------|---------|

# | Python 3.13 | Core language |

# | TensorFlow/Keras | CNN Model |

# | OpenCV | Real-time video |

# | NumPy | Data processing |

# | Matplotlib | Visualization |

# | WinSound | Alert system |

# 

# \## 📁 Project Structure

# ```

# drowsiness\_detection/

# ├── dataset/

# │   ├── open/          # Open eye images

# │   └── closed/        # Closed eye images

# ├── model/

# │   └── drowsiness\_model.h5

# ├── train\_model.py

# ├── realtime\_detect.py

# ├── detect\_simulation.py

# ├── check\_model.py

# ├── create\_test\_video.py

# ├── training\_results.png

# └── README.md

# ```

# 

# \## 🧠 Model Architecture

# ```

# Input (80x80 grayscale)

# &nbsp;   ↓

# Conv2D(32) + BatchNorm + MaxPool + Dropout

# &nbsp;   ↓

# Conv2D(64) + BatchNorm + MaxPool + Dropout

# &nbsp;   ↓

# Conv2D(128) + BatchNorm + MaxPool + Dropout

# &nbsp;   ↓

# Dense(256) + Dropout

# &nbsp;   ↓

# Dense(128) + Dropout

# &nbsp;   ↓

# Dense(1) sigmoid → Open/Closed

# ```

# 

# \## 📊 Model Performance

# | Metric | Value |

# |--------|-------|

# | Training Accuracy | 99.62% |

# | Validation Accuracy | 96.73% |

# | Early Stopping Epoch | 16 |

# | Best Epoch | 9 |

# | Dataset Size | 48,000 images |

# 

# \## ⚙️ Installation

# 

# \### 1. Clone repository

# ```bash

# git clone https://github.com/AkashMS-2005/drowsiness-detection.git

# cd drowsiness-detection

# ```

# 

# \### 2. Create virtual environment

# ```bash

# python -m venv drowsy\_env

# drowsy\_env\\Scripts\\activate

# ```

# 

# \### 3. Install dependencies

# ```bash

# pip install tensorflow opencv-python numpy matplotlib scikit-learn imutils pygame pillow scipy

# ```

# 

# \### 4. Download Dataset

# Download from Kaggle:

# ```

# https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd

# ```

# Place images in `dataset/open/` and `dataset/closed/`

# 

# \## 🚀 Usage

# 

# \### Train Model

# ```bash

# python train\_model.py

# ```

# 

# \### Run Real-Time Detection

# ```bash

# python realtime\_detect.py

# ```

# 

# \### Run Simulation

# ```bash

# python detect\_simulation.py

# ```

# 

# \## 🎮 Controls

# | Key | Action |

# |-----|--------|

# | SPACE | Pause/Resume |

# | Q | Quit |

# 

# \## 🔍 How It Works

# 1\. Webcam captures real-time video frames

# 2\. Face detected using Haar Cascade

# 3\. Eye regions extracted mathematically from face

# 4\. CNN model predicts open/closed for each eye

# 5\. Drowsiness score calculated from predictions

# 6\. Alert fires when score exceeds threshold

# 

# \## 📈 Training Results

# !\[Training Results](training\_results.png)

# 

# \## ⚠️ Alert System

# \- 🟢 AWAKE — Eyes open detected

# \- 🔴 DROWSY — Eyes closed detected

# \- 🚨 ALERT — Beep sound + Red border when drowsy for too long

# 

# \## 👨‍💻 Author

# \*\*Akash M S\*\*

# \- GitHub: \[@AkashMS-2005](https://github.com/AkashMS-2005)

# \- Email: akashmsaiml@gmail.com

# 

# \## 📄 License

# This project is open source and available under the MIT License.

