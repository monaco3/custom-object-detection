import subprocess
import sys

#For Self installation of packages, comes in handy While running on amazon notebook terminal
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

#Packages to be installed
# install("boto3")
# install("opencv-python")

import boto3
import cv2
import threading
from datetime import datetime
from datetime import timedelta

#Seting at what point in time to start. For example start 30 sec,mins,hrs ago. Used to set when to start reading streams
starting_time = datetime.utcnow() - timedelta(seconds=30)

#function to grab kinesis video streams and return the respective HLS video URLs/Links
def my_streams (previewName,streams):
    client = boto3.client('kinesisvideo')
    boto3.setup_default_session(region_name="eu-west-1")
    response = client.get_data_endpoint(
    StreamName=streams,
    APIName='GET_HLS_STREAMING_SESSION_URL'
)
    print(response)
    endpoint = response.get('DataEndpoint', None)
    print("endpoint %s" % endpoint)

    if endpoint is None:
        raise Exception("endpoint none")

    if endpoint is not None:
        client2 = boto3.client('kinesis-video-archived-media', endpoint_url=endpoint)
        url = client2.get_hls_streaming_session_url(
        StreamName=streams,
        PlaybackMode="LIVE_REPLAY",
        HLSFragmentSelector={
            "TimestampRange":{
                "StartTimestamp": datetime.strptime('2020-12-14 06:00:00.243860', '%Y-%m-%d %H:%M:%S.%f') #Set to a specific time
                #set to some some secs,m or hrs ago
                # "StartTimestamp": starting_time
        }
    }
     )['HLSStreamingSessionURL']
        print(url)
        return url

#To visualize/write the captured frames
#         vcap = cv2.VideoCapture(url)
#
#         #ret, frame = vcap.read()
#         frame_width = int(vcap.get(3))
#         frame_height = int(vcap.get(4))
#
#         size = (frame_width, frame_height)
#
#Writing the video frames into video files
#         result = cv2.VideoWriter('capturedtestvideo'+str(previewName)+'.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, size)
#
#
#         while(True):
#         # Capture frame-by-frame
#             ret, frame = vcap.read()
#         #
#             if frame is not None:
#              result.write(frame)
#             #Display the resulting frame
#              cv2.imshow(previewName,frame)
#
#             # Press q to close the video windows before it ends if you want
#             if cv2.waitKey(22) & 0xFF == ord('q'):
#                      break
#             #else:
#             #    print("Frame is None")
#             #    break
#
# # When everything is done, release the capture
#         vcap.release()
#         cv2.destroyAllWindows()
#         print("Video stop")
#
# # t1 = threading.Thread(target=my_streams, args=("second_camera", stream_name_1))
# # t2 = threading.Thread(target=my_streams, args=("main_camera", stream_name_2))
# # t1.start()
# # t2.start()
