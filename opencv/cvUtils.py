import cv2


def destroy_all_windows():
    """Destory all windows."""
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        print('destory all of the window')
        cv2.destroyAllWindows()
        
        
def cv_show(name, img):
    """Show a figure with OpenCV."""
    cv2.imshow(name, img)
    destroy_all_windows()
    
    