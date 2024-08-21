# Rebar Count

## Overview
Rebar Count is a computer vision project designed to detect and count rebar in construction images. The project leverages FRCNN-based models and uses Python for data processing, evaluation, and deployment. The output images show the detected rebar with bounding boxes and the total count.

## Project Structure
The project directory is organized as follows:

```

Rebar_Count/
│
├── coco_eval.py                 # Evaluation script using COCO metrics
├── coco_utils.py                # Utility functions for COCO evaluation
├── Data/                        # Directory containing input data scripts
│   └── import_data.py           # Script to import data
├── steps_for_results/                        # Directory containing demo images
│   ├── 1.png                    # Example input image 1
│   ├── 2.png                    # Example input image 2
│   ├── 3.png                    # Example input image 3
│   └── 4.png                    # Example input image 4
├── engine.py                    # Script for running the detection engine
├── gradio.py                    # Script for building a Gradio interface for the model
├── README.md                    # Project documentation
├── Rebar_Count.ipynb            # Jupyter notebook for model development and experimentation
├── Requirements.txt             # List of required Python libraries
├── Results/                     # Directory containing result images
│   ├── Unknown.png              # Output result image 1
│   ├── Unknown_2.png            # Output result image 2
│   ├── Unknown_3.png            # Output result image 3
│   └── Unknown_4.png            # Output result image 4
├── tests/                       # Directory containing test images
│   └── 0C006B5C.jpg             # Example test image
├── transforms.py                # Script for handling image transformations
└── utils.py                     # Utility functions for the project

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
   git clone https://github.com/yourusername/Rebar_Count.git
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
