# Goal
* Learn to draw different geometric shapes with OpenCV
* You will learn these functions : cv2.line(), cv2.circle() , cv2.rectangle(), cv2.ellipse(), cv2.putText() etc.
# Code
 * img: The image where you want to draw the shapes
 * color: Color of the shape. for BGR, pass it as a tuple, eg: (255,0,0) for blue. For grayscale, just pass the scalar value.
 * thickness: Thickness of the line or circle etc. If **-1** is passed for closed figures like circles, it will fill the shape. default thickness = 1
 * lineType: Type of line, whether 8-connected, anti-aliased line etc. By default, it is *** 8-connected ***. `cv2.LINE_AA` gives anti-aliased line which looks great for curves.