{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "59fe7207-ff2b-459f-9c18-573e9657810e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "\n",
    "# Function to draw bounding box on frame\n",
    "def draw_bounding_box(frame, bbox):\n",
    "    x, y, w, h = bbox\n",
    "    # Draw a rectangle (or oval) around the bounding box\n",
    "    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)\n",
    "    return frame\n",
    "\n",
    "# Main function to process video frames\n",
    "def process_video(video_source):\n",
    "    cap = cv2.VideoCapture(video_source)\n",
    "    if not cap.isOpened():\n",
    "        print(\"Error: Unable to open video source.\")\n",
    "        return\n",
    "\n",
    "    while True:\n",
    "        ret, frame = cap.read()\n",
    "        if not ret:\n",
    "            print(\"Error: Unable to read frame.\")\n",
    "            break\n",
    "\n",
    "        # Example bounding box coordinates (x, y, width, height)\n",
    "        bbox = (100, 100, 200, 150)\n",
    "\n",
    "        # Draw bounding box on the frame\n",
    "        frame_with_bbox = draw_bounding_box(frame.copy(), bbox)\n",
    "\n",
    "        # Display the frame with bounding box\n",
    "        cv2.imshow(\"Video Feed\", frame_with_bbox)\n",
    "\n",
    "        # Break loop if 'q' key is pressed\n",
    "        if cv2.waitKey(1) & 0xFF == ord('q'):\n",
    "            break\n",
    "\n",
    "    cap.release()\n",
    "    cv2.destroyAllWindows()\n",
    "\n",
    "# Run the main function with the video source (e.g., webcam index or video file path)\n",
    "# Main script execution\n",
    "if __name__ == '__main__':\n",
    "    process_video(0)  # Example: Use webcam as video source; Change the argument to specify the video source (0 for webcam)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "imageclassification",
   "language": "python",
   "name": "imageclassification"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
