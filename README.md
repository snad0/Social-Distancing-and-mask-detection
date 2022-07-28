# Social-Distancing-and-mask-detection

To create this, I first broke the problem into 2 halves. The first was to create an app to detect masks and the second was to detect social distancing.
For the first part, I had to create a custom trained model for mask detection. To do so, I used the dataset available on Kaggle. Then convert this data into an array and after splitting the dataset into training and test data and then performing the data augmentation. I trained the deep convolutional model on training data after 20 epochs when the accuracy was around 98 percent.For better accuracy in less iteration i used transfer learning approach to build the model after that i saved that model to use it further.


To apply my mask detection, I also needed a face detection model. For that, I’ve used a pretrained face detector named res10_300x300_ssd_iter_140000 from OpenCV. This is the DNN for face detection. Then I applied a prediction function to our webcam using cv2. This function first applies a face detector to the video frame, detecting faces using it. After detecting faces, it applies our mask model to those faces and gives us the results. The accuracy was very good while using web cam, but with videos due to the low quality, the accuracy was somehow decreasing but was still good. We can increase this by using a better and larger dataset.
The second part includes three steps: object detection, object tracking, and distance measurement.


The object detection was done by YOLO. It is an algorithm that uses neural networks to provide real-time object detection. This algorithm is popular because of its speed and accuracy. It has been used in various applications to detect traffic signals, people, animals, etc. This is the best object detection model to my knowledge, as it can detect over 9000 classes.


Object tracking is done by drawing boxes over the object and assigning a new id to each object. Then we track the position of that centroid in each frame. The distance measurement part is taken care of by using a centroid, which is the centre point of any object we have detected. Then we find the Euclidean distance between those centroids, having a fixed minimum value for the social distancing part.

As similar to the first part, I applied yolo object detection on a video using openCV video capture, then drew boxes over each object, defining its position and then using these positions to find the centroid. Then, if the number of objects in a frame is greater than one, I calculate the distance between all the centroids, which is used to determine if there are any violations of social distancing or not. The accuracy of yolo was as good as expected, so I merged both these steps to create the final project. The results were satisfactory, and the accuracy was also fine but can be improved.



Input video

https://user-images.githubusercontent.com/93977986/180812624-57c46988-308f-4a18-910d-7beb41c5c8aa.mp4


Output video


https://user-images.githubusercontent.com/93977986/180814013-16679434-93cb-41ce-b3bb-22f9e7825971.mp4





https://user-images.githubusercontent.com/93977986/180814150-c6412388-4005-4f90-a1bc-873069913d3e.mp4






https://user-images.githubusercontent.com/93977986/180816304-5d734e81-f50f-4a9a-a8f2-558d5df09c78.mp4


