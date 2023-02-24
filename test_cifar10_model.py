import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import unittest
import torch.autograd as autograd
import matplotlib.pyplot as plt


class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        return x


''' Unit tests for the CIFAR10Model class '''
''' Run this file from the command line with: python3 -m unittest test_cifar10_model.py '''


class TestCIFAR10Model(unittest.TestCase):

    def test_model_performance(self):
        # Test that the model's performance is consistent across different hyperparameters
        batch_sizes = [32, 64, 128]
        learning_rates = [0.0001, 0.001, 0.01]
        num_epochs = [5, 10, 15]

        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                for n_epochs in num_epochs:

                    print(
                        f"Training with batch_size={batch_size}, learning_rate={learning_rate}, num_epochs={n_epochs}")

                    train_dataset = torchvision.datasets.CIFAR10(root='path/to/data', train=True,
                                                                 download=True, transform=transforms.ToTensor())
                    test_dataset = torchvision.datasets.CIFAR10(root='path/to/data', train=False,
                                                                download=True, transform=transforms.ToTensor())
                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                               shuffle=True)
                    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                              shuffle=False)
                    model = CIFAR10Model()
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=learning_rate)

                    train_losses = []
                    train_acc = []
                    test_acc = []

                    for epoch in range(n_epochs):

                        print(f"Training epoch {epoch + 1} of {n_epochs}")

                        for i, (images, labels) in enumerate(train_loader):
                            # Forward pass
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                            # Backward and optimize
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            train_losses.append(loss.item())

                        with torch.no_grad():

                            # Train the model
                            correct_train = 0
                            total_train = 0
                            correct_test = 0
                            total_test = 0
                            for images, labels in train_loader:
                                outputs = model(images)
                                _, predicted = torch.max(outputs.data, 1)
                                total_train += labels.size(0)
                                correct_train += (predicted ==
                                                  labels).sum().item()
                            train_acc.append(100 * correct_train / total_train)

                            # Test the model
                            correct = 0
                            total = 0
                            for images, labels in test_loader:
                                outputs = model(images)
                                _, predicted = torch.max(outputs.data, 1)
                                total += labels.size(0)
                                correct += (predicted == labels).sum().item()
                            test_acc.append(100 * correct / total)

                            # Calculate bias and variance
                            bias = 100 - test_acc[-1]
                            variance = max(train_acc) - test_acc[-1]
                            print(
                                f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_acc[-1]:.2f}%, Test Accuracy: {test_acc[-1]:.2f}%, Bias: {bias:.2f}%, Variance: {variance:.2f}%')

                    # Plot the train and test accuracy, bias, and variance over time
                    plt.plot(train_acc, label='Train accuracy')
                    plt.plot(test_acc, label='Test accuracy')
                    plt.plot([0, n_epochs], [100, 100], linestyle='--',
                             color='gray', label='Chance level')
                    plt.plot([0, n_epochs], [100-bias, 100-bias],
                             linestyle='--', color='red', label='Bias')
                    plt.plot([0, n_epochs], [100-variance, 100-variance],
                             linestyle='--', color='blue', label='Variance')
                    plt.xlabel('Epoch')
                    plt.ylabel('Accuracy')
                    plt.legend()
                    plt.show()

                    # chance level for CIFAR10 is 10%
                    self.assertGreater(test_acc[-1], 10)
                    self.assertLess(train_losses[-1], train_losses[0])

    def test_model_output_shape(self):
        # Test that the output of the model has the expected shape
        input_data = torch.rand(self.batch_size, 3, 32, 32)
        output = self.model(input_data)
        self.assertEqual(output.shape, (self.batch_size, 10))

    def test_model_trainable_parameters(self):
        # Test that the model has trainable parameters
        self.assertGreater(len(list(self.model.parameters())), 0)

    def test_model_train_loss(self):
        # Test that the training loss decreases over time
        train_losses = []
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

        self.assertLess(train_losses[-1], train_losses[0])

    def test_model_test_accuracy(self):
        # Test that the model achieves a test accuracy greater than chance level
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_acc = 100 * correct / total
        self.assertGreater(test_acc, 10)  # chance level for CIFAR10 is 10%

    def test_model_train_accuracy(self):
        # Test that the training accuracy improves over time
        for epoch in range(self.num_epochs):
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in self.train_loader:
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                train_acc = 100 * correct / total
                self.assertTrue(len(self.train_acc) ==
                                0 or train_acc >= self.train_acc[-1])
                self.train_acc.append(train_acc)

    def test_model_train_test_comparison(self):
        # Test that the test accuracy is not significantly worse than the training accuracy
        for epoch in range(self.num_epochs):
            with torch.no_grad():
                correct_train = 0
                total_train = 0
                correct_test = 0
                total_test = 0

                for images, labels in self.train_loader:
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()
                train_acc = 100 * correct_train / total_train

                for images, labels in self.test_loader:
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()
                test_acc = 100 * correct_test / total_test

            # the test accuracy should not be significantly worse than the train accuracy
            self.assertGreaterEqual(train_acc, test_acc)

    def test_model_predictions(self):
        # Test that the model's predictions are meaningful
        input_data = torch.rand(self.batch_size, 3, 32, 32)
        true_labels = torch.randint(low=0, high=10, size=(self.batch_size,))
        with torch.no_grad():
            outputs = self.model(input_data)
            _, predicted = torch.max(outputs.data, 1)
            self.assertTrue(torch.equal(predicted, true_labels))

    def test_gradients(self):
        # Generate random input and labels
        input_data = torch.randn(self.batch_size, 3, 32, 32)
        labels = torch.randint(low=0, high=10, size=(self.batch_size,))

        # Compute gradients using the PyTorch autograd function
        inputs = input_data.requires_grad_(True)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        grads = autograd.grad(loss, self.model.parameters(), create_graph=True)

        # Check the gradients using the PyTorch gradcheck function
        for grad in grads:
            self.assertTrue(autograd.gradcheck(
                lambda x: autograd.grad(loss, x, create_graph=True), grad))

    def test_model_resizing_input(self):
        # Test that the model can handle input images of different sizes
        input_data1 = torch.rand(self.batch_size, 3, 32, 32)
        output1 = self.model(input_data1)
        self.assertEqual(output1.shape, (self.batch_size, 10))

        input_data2 = torch.rand(self.batch_size, 3, 64, 64)
        output2 = self.model(input_data2)
        self.assertEqual(output2.shape, (self.batch_size, 10))

        input_data3 = torch.rand(self.batch_size, 3, 128, 128)
        output3 = self.model(input_data3)
        self.assertEqual(output3.shape, (self.batch_size, 10))


if __name__ == '__main__':
    unittest.main()
