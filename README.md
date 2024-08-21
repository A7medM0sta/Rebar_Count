# Rebar Count

## Overview
Rebar Count is a computer vision project designed to detect and count rebar in construction images. The project leverages FRCNN-based models and uses Python for data processing, evaluation, and deployment. The output images show the detected rebar with bounding boxes and the total count.

## Project Structure
The project directory is organized as follows:

```

Rebar_Count/
│
├── notebooks/                    # Jupyter notebooks for experimentation and development
│   └── Rebar_Count.ipynb
│
├── src/                          # Source code for the project
│   ├── __init__.py               # Initialize the src module
│   ├── engine.py                 # Main engine for running the detection model
│   ├── gradio_app.py             # Gradio interface for the model
│   ├── transforms.py             # Image transformation functions
│   ├── utils.py                  # General utility functions
│   ├── evaluation/               # Scripts for evaluation metrics
│   │   ├── coco_eval.py          # COCO evaluation script
│   │   └── coco_utils.py         # Utility functions for COCO evaluation
│   └── data/                     # Data-related scripts
│       └── import_data.py        # Script to import data
│
├── data/                         # Directory for data (images, datasets, etc.)
│   ├── demo/                     # Demo images for testing
│   │   ├── 1.png
│   │   ├── 2.png
│   │   ├── 3.png
│   │   └── 4.png
│   └── tests/                    # Test images for validation
│       └── 0C006B5C.jpg
│
├── results/                      # Directory to store model output (results)
│   ├── Unknown.png
│   ├── Unknown_2.png
│   ├── Unknown_3.png
│   └── Unknown_4.png
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── LICENSE                       # License file (if applicable)

```

## Requirements
The project requires the following Python libraries. You can install them using `pip`:

```bash
pip install -r Requirements.txt
```

The key dependencies include:

- PyTorch
- YOLOv5
- OpenCV
- Gradio

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com//Rebar_Count.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Rebar_Count
   ```

3. Install the required dependencies:

   ```bash
   pip install -r Requirements.txt
   ```

## Usage

### Running the Jupyter Notebook

To explore and test the model, open the `Rebar_Count.ipynb` notebook in Jupyter:

```bash
jupyter notebook Rebar_Count.ipynb
```

### Running the Detection Engine

You can run the detection engine on your images by executing the `engine.py` script:

```bash
python engine.py --input_path /path/to/your/images --output_path /path/to/save/results
```

### Gradio Interface

To launch the Gradio interface for a user-friendly web application, run the `gradio.py` script:

```bash
python gradio.py
```

## Results

The results from running the model are saved in the `Results/` directory. Each image is processed and saved with detected rebar highlighted by bounding boxes and the count displayed.

## Testing

To test the model on sample images, place your images in the `tests/` directory and run the detection engine as described above.

## Future Work

- Improving the accuracy of rebar detection by experimenting with different models and techniques.
- Expanding the project to detect other construction materials and objects.
- Optimizing the Gradio interface for better user experience.

## Contributing

If you would like to contribute to this project, please open a pull request or create an issue to discuss your ideas.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

```

### Notes:
1. **Repository URL:** Replace `https://github.com/yourusername/Rebar_Count.git` with your actual GitHub repository link.
2. **Future Work:** Modify or add more details based on your project goals.
3. **Contributing and License:** If these sections are relevant to your project, feel free to keep them. If not, you can remove them.
