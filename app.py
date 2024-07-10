{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "00e2f5d8-5dbe-4a67-b355-538619e25af1",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, request, jsonify\n",
    "from bounding_box import process_video\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "# Load the trained model and start video processing on application startup\n",
    "def startup():\n",
    "    process_video(0)  # Example: Use webcam as video source\n",
    "\n",
    "# Main script execution\n",
    "if __name__ == '__main__':\n",
    "    startup()\n",
    "    app.run(host='0.0.0.0', port=5000)"
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
