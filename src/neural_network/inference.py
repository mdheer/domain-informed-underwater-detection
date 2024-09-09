import torch

from neural_network.utils import (
    MeasuredDataset,
    DomainKnowledgeDataset,
    SimpleFNN,
)


def perform_inference(data):
    # Get model architecture
    model = SimpleFNN(input_size=3998, num_classes=2)

    # Load the trained weights
    model.load_state_dict(
        torch.load(r"thesis_mathieu\_4_nn\trained_models\best_model.pth")
    )

    # Put the model in evaluation stage
    model.eval()

    # Apply model to data
    logits = model(data)

    # Apply softmax to output to get probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1)

    # Choose the class with the highest probability
    predicted_class = torch.argmax(probabilities, dim=1)

    print(predicted_class)
