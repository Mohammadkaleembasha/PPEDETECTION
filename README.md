Technologies Used:
•	YOLOv5 (You Only Look Once): A cutting-edge deep learning model for real-time object detection.
•	Ultralytics: A library for training YOLO models.
•	OpenCV: For video processing and frame analysis.
•	Python: Programming language for implementing the model.
•	Deep Learning (CNN): The underlying architecture that enables accurate object detection in real-time.
________________________________________
Project Description:
1. Data Collection and Annotation:
To build a reliable detection model, I started by gathering a diverse dataset of images containing people wearing and not wearing PPE. These images were sourced from public datasets and workplace environments, ensuring different conditions like lighting and angles were represented. The next step was to annotate these images using a tool called LabelImg. I manually drew bounding boxes around PPE items, such as hardhats, masks, and safety vests, and labeled each object accordingly. This annotation created a labeled dataset that was vital for training the model.
________________________________________
2. Model Training with YOLOv5:
Once the dataset was ready, I moved to the model training phase using YOLOv5, which is known for its high-speed performance in real-time applications. YOLOv5 divides each image into a grid and predicts bounding boxes and class probabilities for each grid cell.
•	Pre-trained model: To expedite the training process, I started with a pre-trained YOLOv5 model from Ultralytics. This gave me a strong baseline since the pre-trained model already knew how to detect general objects.
•	Customizing the model: I adapted the model to detect specific PPE items by fine-tuning it on my annotated dataset. The dataset was split into training and validation sets, with the training set used to teach the model and the validation set to test its performance.
•	Hyperparameter Tuning: I adjusted key hyperparameters such as the learning rate, batch size, and number of epochs to optimize performance. Learning rate controls how much the model’s weights are updated during training, while batch size determines how many samples the model processes before updating. I experimented with these values to improve both accuracy and training speed.
•	Data Augmentation: To make the model robust to variations in lighting, angles, and environments, I applied data augmentation techniques like rotation, flipping, and scaling. This helped prevent overfitting and ensured the model performed well on unseen data.
•	Loss Function: The training process focused on minimizing the object detection loss, which comprises classification loss, bounding box loss, and confidence loss. By fine-tuning these aspects, I was able to improve the model’s detection accuracy.
•	Training Time: The model was trained over 50 epochs, and I monitored its performance using the validation set after each epoch. The final model was able to achieve an accuracy of 79% in detecting PPE items such as hardhats, masks, and safety vests.
After training, the model was saved as ppe.pt, ready to be deployed for real-time detection.
________________________________________
3. Real-time Detection with OpenCV:
To bring the model into action, I integrated it with OpenCV for real-time video processing. OpenCV allowed me to capture video from various sources like video files, webcams, and external cameras.
•	Frame-by-frame processing: For each video frame, the system passed it through the YOLO model, which predicted bounding boxes around detected PPE items. The system was designed to display a confidence score for each detected object.
•	Color-coded alerts: To make the system intuitive, I implemented a color-coded alert mechanism:
o	Green for compliant workers (e.g., wearing hardhats, masks).
o	Red for non-compliant workers (e.g., no hardhat or mask detected).
This real-time detection system ensures that safety officers can quickly identify workers who are not following safety protocols.
________________________________________
Challenges and Solutions:
•	Real-time Performance: One of the key challenges was ensuring that the system could process high-resolution video in real-time without lag. To address this, I optimized the code by resizing video frames before processing and using efficient data structures to handle bounding box calculations. This improved the system’s processing speed.
•	Detection Accuracy: Achieving a high level of detection accuracy in varying environments (e.g., low light or cluttered backgrounds) was another challenge. I addressed this by fine-tuning the model with a well-labeled and diverse dataset, as well as experimenting with different versions of YOLO.
________________________________________
Impact and Results:
The system was tested in various workplace environments such as construction sites. It successfully identified workers who were not wearing the required PPE (e.g., hardhats, masks) and provided real-time alerts to safety officers. The system’s ability to process video feeds in real-time with an accuracy of 79% significantly enhances workplace safety by ensuring compliance with PPE standards.
________________________________________
Future Improvements:
In the future, I plan to expand the system’s capabilities by including additional PPE items like gloves and safety glasses. Furthermore, I aim to enhance the model’s accuracy by training it on a more diverse and larger dataset. I am also considering developing a web-based interface for remote monitoring, allowing safety officers to track compliance across multiple locations simultaneously and receive alerts in real-time.
