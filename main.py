import threading
import pca_implement as pca

user = ""

image_thread = threading.Thread(target = pca.run_pca, args=())
gui_thread = threading.Thread(target = gui_update, args=())

image_thread.start()
gui_thread.start()
image_thread.join()
gui_thread.join()
