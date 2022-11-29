## Progress Update
### Nov 28th

#### Progress

1. Constructed Calculators for different levels of facial information. (See FacialExpression.md)
2. Learning Filtergraph and constructed filtergraph collections towards different emotions. (See filterGraphEmotion.maxpat)
3. I worked with the facial landmark recognition model in MAX a bit and I would like to compare more of whether to conduct the detection task in Python or in MAX. (See face_test.maxpat)
4. I created a video feedback exploration using Vizzie so the calculated would influence both the audio and video playback. (See face_video.maxpat)
5. For my previous work in the last quarter, the codes are in the "previous" branch for you to compare.

#### Question
1. I haven't figured out the blob sending feature in OSC in MAX. I can however send the pixel value through OSCMessage and the message can be read by MAX. But when sending OCSBlob, the pyOSC3 works while the receiving part in MAX is not working.
2. The FacialExpression.md file is used for creating meaningful mapping, and it should be notable and easy to reproduce for audiences.
3. Still I haven't solve the multiple faces part.

#### Next step for the coming class.
1. Working with filtergraphs. Controlling the sound effect using emotions.
2. Working with Facial Expression Features. Controlling pitch, rate, etc using mouth width and face position.
