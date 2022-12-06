import pyOSC3
import numpy as np
import math
import struct

client = pyOSC3.OSCClient()
client.connect( ( 'localhost', 3000 ) )

while True:
    random_3d_matrix_array = 255*np.random.rand(60, 60);
    b = random_3d_matrix_array.flatten().astype("uint8").tobytes()


    msg = pyOSC3.OSCBundle()
    msg.setAddress("OSCBlob")
    msg.append(b,typehint = 'b')
    client.send(msg)
