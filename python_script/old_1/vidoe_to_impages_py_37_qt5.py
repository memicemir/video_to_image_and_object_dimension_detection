# MIT License

# Copyright (c) 2023 Emir Memic

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

print ('hello me')

import os, sys, glob
import cv2

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import *

from PIL import Image, ImageDraw 

import numpy as np
import imutils
from imutils import perspective
from imutils import contours

from scipy.spatial import distance as dist

from mp4_to_jpg_gui_mask_qt_5 import Ui_MainWindow

import matplotlib.pyplot as plt

print ('works?')

class WindowInt(QMainWindow):

    def __init__(self, parent = None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
       
        print ('hello me again')
        print ('...')
        
        print ('hello mou')
        
        self.ui.pushButton.clicked.connect(self.convert_mp4_to_jpg)
        self.ui.pushButton_2.clicked.connect(self.list_available_mp4_videos)
        self.ui.pushButton_3.clicked.connect(self.find_edges)

        self.ui.pushButton_4.clicked.connect(self.full_auto_lenght_width)

        self.ui.checkBox_2.setChecked(True)
        self.ui.checkBox_2.clicked.connect(self.give_values)
        self.ui.checkBox_3.clicked.connect(self.mean_values)
        self.ui.checkBox_4.clicked.connect(self.median_values)
        
        #self.ui.checkBox_6.clicked.connect(self.rgb_to_hsv)
        
        #self.ui.checkBox_11.clicked.connect(self.rgb_to_gray)		

        #self.ui.checkBox_5.setChecked(True)
        self.ui.checkBox_7.setChecked(True)
        self.ui.checkBox_8.setChecked(True)
        #self.ui.checkBox_9.setChecked(True)
        self.ui.checkBox_14.setChecked(True)
        self.ui.checkBox_12.setChecked(True)
        
        self.ui.checkBox_15.setChecked(True)
        self.ui.checkBox_17.setChecked(True)
        #self.ui.checkBox_20.setChecked(True)
        
        #self.ui.checkBox_21.setChecked(True)        

        #self.ui.checkBox_16.setEnabled(False)
        #self.ui.checkBox_17.setEnabled(False)

        try:
            os.mkdir('./IntermediateOutputs')
        except:
            print ('./IntermediateOutputs directory exists!')
            
        try:
            os.mkdir('./Outputs')
        except:
            print ('./Outputs - for assembled image and analysed directory exists!')	

        try:
            os.mkdir('./BackupImages')
        except:
            print ('./BackupImages - for assembled image and analysed directory exists!')
##############
        intermediateDirContRemoveIntermediateOut = glob.glob('./IntermediateOutputs/*.jpg')
        for AllintOut in intermediateDirContRemoveIntermediateOut:
            try:
                os.remove(AllintOut)
                print('All files removed from intermediate directory!')
            except:
                print('Removal intermediate out dir exception hit!')				
####################
        intermediateDirContRemoveGeneralOut = glob.glob('./Outputs/*.*')
        for AllGenOut in intermediateDirContRemoveGeneralOut:
            try:
                os.remove(AllGenOut)
                print('All files removed from general output directory!')
            except:
                print('Removal general output dir exception hit!')
#######################
        intermediateDirContRemoveImageBackup = glob.glob('./BackupImages/*.jpg')
        for AllImBack in intermediateDirContRemoveImageBackup:
            try:
                os.remove(AllImBack)
                print('All files removed from image backup directory!')
            except:
                print('Removal image backup dir exception hit!')
#################################                

        # on program start clean files to prevent overflow of appending ...
        #full lenght
        # with open('./Outputs/' + 'Full_auto_lenght_widht.txt', 'w') as writeFullOut:
            # writeFullOut.write('PlotNo' + '\t\t' + 'LeafNo' + '\t\t' + 'RankNo' + '\t\t' + 'SegNo' + '\t\t' +  'Lenght' + '\t\t' +  'Width' + '\n')
        # #segmented lenght    
        # with open('./Outputs/' + 'Segmented_auto_lenght_widht.txt', 'w') as writeSegOut:
            # writeSegOut.write('PlotNo' + '\t\t' + 'LeafNo' + '\t\t' + 'RankNo' + '\t\t' + 'SegNo' + '\t\t' +  'Lenght' + '\t\t' +  'Width' + '\n')

        with open('./Outputs/' + 'Full_skelet_values.txt', 'w') as writeSkeletFull:
            writeSkeletFull.write(
            '{:>10}'.format('PlotNo') + '\t' 
            + '{:>10}'.format('LeafNo') + '\t' 
            + '{:>10}'.format('RankNo') + '\t' 
            + '{:>10}'.format('SegNo') + '\t'        
            +  str('LnghtSumLeft') + '\t'
            +  str('LnghtSum') + '\t'
            +  str('LnghtSumRight') + '\t'            
            +  str('diamter')                        
            + '\n')             
        with open('./BackupImages/' + 'Full_skelet_values.txt', 'w') as writeSkeletFullBackup:
            writeSkeletFullBackup.write(
            '{:>10}'.format('PlotNo') + '\t' 
            + '{:>10}'.format('LeafNo') + '\t' 
            + '{:>10}'.format('RankNo') + '\t' 
            + '{:>10}'.format('SegNo') + '\t'        
            #+  str('LnghtSumLeft') + '\t'
            +  str('LnghtSum') + '\t'
            #+  str('LnghtSumRight') + '\t'            
            +  str('diamter')                        
            + '\n')             


                
    #def rgb_to_hsv(self):
        #self.ui.checkBox_6.setChecked(True)
        #self.ui.checkBox_11.setChecked(False)		
    def rgb_to_gray(self):
        #self.ui.checkBox_6.setChecked(False)
        self.ui.checkBox_11.setChecked(True)			

    def give_values(self):
        self.ui.checkBox_2.setChecked(True)
        self.ui.checkBox_3.setChecked(False)
        self.ui.checkBox_4.setChecked(False)
        
    def mean_values(self):
        self.ui.checkBox_2.setChecked(False)
        self.ui.checkBox_3.setChecked(True)
        self.ui.checkBox_4.setChecked(False)
                
    def median_values(self):
        self.ui.checkBox_2.setChecked(False)
        self.ui.checkBox_3.setChecked(False)
        self.ui.checkBox_4.setChecked(True)				

    def list_available_mp4_videos(self):

        selectDir = QFileDialog.getExistingDirectory()
        print ('selectDir', selectDir)
        self.ui.lineEdit.setText(selectDir)
        
        listOfImagesAnalysis = []
        #for images in glob.glob1(selectDir, '*.mp4'):
        for images in glob.glob1(selectDir, '*'):	
            listOfImagesAnalysis.append(images)
        self.ui.listWidget.addItems(listOfImagesAnalysis)	


    def convert_mp4_to_jpg(self):

        pathToMp4Video = str(self.ui.lineEdit.text())

        for selectedMp4 in self.ui.listWidget.selectedItems():
            videoUse = str(selectedMp4.text())
            QApplication.processEvents()
            #self.imgNameOutput = str(selectedIMG.text()).split('.')[0]
            if str(videoUse.split('.')[-1]) == 'mp4' or str(videoUse.split('.')[-1]) == 'MP4':
            #if str(videoUse.split('.')[1]) == 'mp4':	
                cam = cv2.VideoCapture(pathToMp4Video + '/' + videoUse)
                #cam = cv2.VideoCapture('video.mp4')
                #cam = cv2.VideoCapture('Maize_LeafShape_220721.mp4')


                length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
                print( length )

                currentFrame = 0

                saveFrameCounter = 0

                endCount = 0

                    
                while True:
                    ret, frame = cam.read()
                    QApplication.processEvents()
                    if ret:
                        name = './IntermediateOutputs/' + str(currentFrame) + '.jpg'
                        print ('Creating...' + name)
                        
                        if saveFrameCounter == int(str(self.ui.lineEdit_8.text())): 
                            print ('save countr', saveFrameCounter)
                            
                            cv2.imwrite(name, frame)
                            
                            saveFrameCounter = 0
                            
                            endCount = endCount + 1
                            
                        currentFrame += 1
                        saveFrameCounter += 1
                    #else:
                    if currentFrame == length:
                        break

                cam.release()

                cv2.destroyAllWindows()			

                print ('assambl image')

                start = int(str(self.ui.lineEdit_8.text())) * 2 
                currentImageExtracting = start
                endCount = endCount * int(str(self.ui.lineEdit_8.text())) 

                print ('start', start, 'endCount', endCount)

                for imageCurrent in range(start , endCount + int(str(self.ui.lineEdit_8.text())), int(str(self.ui.lineEdit_8.text()))):
                    #imageCurrent = imageCurrent + 50
                    print ('imageCurrent', imageCurrent )
                    QApplication.processEvents()
                    if imageCurrent == currentImageExtracting: 
                        
                        im1 = Image.open('./IntermediateOutputs/' + str(imageCurrent) + '.jpg')
                        im2 = Image.open('./IntermediateOutputs/' + str(int(imageCurrent - int(str(self.ui.lineEdit_8.text())))) + '.jpg')
                        
                        
                        im1 = im1.crop((0, 231, 1920, 242)) # 9 -> 336
                        im2 = im2.crop((0, 231, 1920, 242))
                        
                        #im1.show()
                        #im2.show()

                    else:
                        #im2 = Image.open('./Outputs/assebled_v.jpg')
                        im2 = Image.open('./Outputs/' + str(videoUse.split('.')[-2]) + '.jpg')
                        im1 = Image.open('./IntermediateOutputs/' + str(int(imageCurrent - int(str(self.ui.lineEdit_8.text())))) + '.jpg')

                        draw = ImageDraw.Draw(im1)
                        
                        # to narrow the line more of kept line
                        if self.ui.checkBox_14.isChecked() and self.ui.checkBox_20.isChecked():
                            draw.line((int(str(self.ui.lineEdit_24.text())), 231, int(str(self.ui.lineEdit_25.text())), 231), width=12)
                        
                        
                        if self.ui.checkBox_14.isChecked() and ((imageCurrent % 2) == 0): # divided by two with left over zero will execute. only even numbers count 
                            #draw = ImageDraw.Draw(im1)
                            #draw.line((0, 280, 1920, 280), width=15)
                            
                            #draw.line((0, 231, 900, 231), width=50)
                            draw.line((int(str(self.ui.lineEdit_24.text())), 231, int(str(self.ui.lineEdit_25.text())), 231), width=100)  
                        
                        
                        im1 = im1.crop((0, 231, 1920, 242))


                        #im1.show('im1')
                        #im2.show('im2')

                    
                    print ('im1', str(imageCurrent), 'im2', str(int(imageCurrent - int(str(self.ui.lineEdit_8.text())))))


                    def get_concat_v(im1, im2):
                        #dst = Image.new('RGB', (im1.width, im1.height + im2.height))
                        #dst.paste(im1, (0, 0))
                        #dst.paste(im2, (0, im1.height))
                        
                        dst = Image.new('RGB', (im1.width, im1.height + im2.height))
                        dst.paste(im2, (0, 0))
                        dst.paste(im1, (0, im2.height))
                        #dst.paste(im1, (0, 0))						
                        
                        return dst		
                        
                    #get_concat_v(im1, im2).save('./Outputs/assebled_v.jpg')
                    get_concat_v(im1, im2).save('./Outputs/' + str(videoUse.split('.')[-2]) + '.jpg')
                        
            if str(videoUse.split('.')[-1]) == 'jpg':
                img = cv2.imread(pathToMp4Video + '/' + videoUse)
                #cv2.imwrite('./Outputs/assebled_v.jpg' , img)
                cv2.imwrite('./Outputs/' + str(videoUse.split('.')[-2]) + '.jpg' , img)        
            


            # if self.ui.checkBox.isChecked():

                # plt.figure(figsize=(10,10))
                # plt.subplot(311)                             #plot in the first cell
                # plt.subplots_adjust(hspace=.5)
                # plt.title("Hue")
                # plt.hist(np.ndarray.flatten(hue), bins=180)
                # plt.subplot(312)                             #plot in the second cell
                # plt.title("Saturation")
                # plt.hist(np.ndarray.flatten(sat), bins=128)
                # plt.subplot(313)                             #plot in the third cell
                # plt.title("Luminosity Value")
                # plt.hist(np.ndarray.flatten(val), bins=128)
                # plt.show()
        
            #
            
            # imageFilGaps = cv2.imread('./Outputs/' + str(videoUse.split('.')[-2]) + '.jpg')
            
            # gray = cv2.cvtColor(imageFilGaps, cv2.COLOR_BGR2GRAY)
            
            # des = cv2.bitwise_not(gray)
            # contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            # for cnt in contour:
                # cv2.drawContours(des,[cnt],0,255,-1)
            # grayOut = cv2.bitwise_not(des)
            
            # # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            # # res = cv2.morphologyEx(imageFilGaps,cv2.MORPH_OPEN,kernel)
            
            # cv2.imwrite('./Outputs/' + str(videoUse.split('.')[-2]) + '_fillGaps.jpg' , grayOut)
            

        
        print ('Finished!')		

    def find_edges(self):

        #full and segmented open to append values lenght
        #with open('./Outputs/' + 'Full_auto_lenght_widht.txt', 'a') as writeFullOut, open('./Outputs/' + 'Segmented_auto_lenght_widht.txt', 'a') as writeSegOut:


        #with open('./Outputs/' + 'Full_auto_lenght_widht.txt', 'a') as writeFullOut:
            #writeFullOut.write('name' + '\t' + 'lenght' + '\n')


        for selectedMp4 in self.ui.listWidget.selectedItems():
            videoUse = str(selectedMp4.text())
            QApplication.processEvents()
            analysingAssembledImage = str(videoUse.split('.')[-2])    

            #writeFullOut.write(analysingAssembledImage)
            try:
                plotNo = analysingAssembledImage.split('-')[-3]
            except:
                plotNo = 'NotA'
            try:        
                leafNo = analysingAssembledImage.split('-')[-2]
            except:
                leafNo = 'NotA'
            try:    
                rankNo = analysingAssembledImage.split('-')[-1]
            except:
                rankNo = 'NotA'
                
            i = 1

            #photoRead = './Outputs/assebled_v.jpg'
            photoRead = './Outputs/' + str(videoUse.split('.')[-2]) + '.jpg'
            print (photoRead)

            target_image = photoRead
            img_0 = cv2.imread(target_image)

            hsv_image = cv2.cvtColor(img_0, cv2.COLOR_BGR2HSV)
            hue = hsv_image[:,:,0]
            sat = hsv_image[:,:,1]
            val = hsv_image[:,:,2]

            mean_hue = np.mean(hue)
            mean_sat = np.mean(sat)
            mean_val = np.mean(val)

            median_hue = np.median(hue)
            median_sat = np.median(sat)
            median_val = np.median(val)

            min_hue = int(np.min(hue))
            min_sat = int(np.min(sat))
            min_val = int(np.min(val))

            max_hue = int(np.max(hue))
            max_sat = int(np.max(sat))
            max_val = int(np.max(val))
             
            print ('hue mean - median:', mean_hue, '-', median_hue)
            print ('hue min - max    :', min_hue,'-', max_hue)
            print ('sat mean - median:', mean_sat, '-', median_sat)
            print ('sat min - max    :', min_sat,'-', max_sat)
            print ('val mean - median:', mean_val, '-', median_val)
            print ('val min - max    :', min_val,'-', max_val)


            if self.ui.checkBox.isChecked():

                plt.figure(figsize=(10,10))
                plt.subplot(311)                             #plot in the first cell
                plt.subplots_adjust(hspace=.5)
                plt.title("Hue")
                plt.hist(np.ndarray.flatten(hue), bins=180)
                plt.subplot(312)                             #plot in the second cell
                plt.title("Saturation")
                plt.hist(np.ndarray.flatten(sat), bins=128)
                plt.subplot(313)                             #plot in the third cell
                plt.title("Luminosity Value")
                plt.hist(np.ndarray.flatten(val), bins=128)
                plt.show()


            if self.ui.checkBox_2.isChecked():
                h_min = int(str(self.ui.lineEdit_2.text()))
                h_upper = int(str(self.ui.lineEdit_3.text()))	
                
                s_min = int(str(self.ui.lineEdit_4.text()))
                s_upper = int(str(self.ui.lineEdit_5.text()))

                v_min = int(str(self.ui.lineEdit_6.text()))	
                v_upper = int(str(self.ui.lineEdit_7.text()))
                    
            if self.ui.checkBox_3.isChecked():	
                h_min = mean_hue -10
                h_upper = mean_hue +10				

                s_min = mean_sat -25	
                s_upper = mean_sat +25	

                v_min = mean_val -25	
                v_upper = mean_val +25	

                #h_min = mean_hue -25
                #h_upper = mean_hue -1				

                #s_min = min_sat	
                #s_upper = mean_sat -1	

                #v_min = min_val		
                #v_upper = max_val

            if self.ui.checkBox_4.isChecked():	
                h_min = median_hue -10
                h_upper = median_hue +10				
                
                s_min = median_sat -25	
                s_upper = median_sat +25

                v_min = median_val -25		
                v_upper = median_val +25	

                #h_min = median_hue -25
                #h_upper = median_hue -1				
                
                #s_min = min_sat	
                #s_upper = median_sat -1	

                #v_min = min_val		
                #v_upper = max_val


            mask = cv2.inRange(hsv_image, (h_min, s_min, v_min), (h_upper, s_upper, v_upper))
            
            whole_image = mask.size
            imask = mask>0
            green = np.zeros_like(img_0, np.uint8)
            green[imask] = img_0[imask]
            green_activ = green[imask].size
            #cv2.imwrite('./Outputs/Filtered_a_' + photoRead, green)
            
            #cv2.imwrite('./Outputs/Filtered.jpg' , green)
            cv2.imwrite('./Outputs/' + str(videoUse.split('.')[-2]) + '_Filtered.jpg' , mask)



        

            #grayImageRead = cv2.imread('./Outputs/try.jpg')
            
            image = cv2.imread('./Outputs/' + str(videoUse.split('.')[-2]) + '_Filtered.jpg')
            
            # image_1 - needed for later to drow on in original image
            image_1 = cv2.imread('./Outputs/' + str(videoUse.split('.')[-2]) + '_Filtered.jpg')
            #image = cv2.imread('./Outputs/DSC.jpg')
            #cv2.imshow("Filtered", image)
            #cv2.waitKey(0)
            
            if self.ui.checkBox_12.isChecked():
                #image = cv2.imread('./Outputs/assebled_v.jpg')
                #image_1 = cv2.imread('./Outputs/assebled_v.jpg')
                image = cv2.imread('./Outputs/' + str(videoUse.split('.')[-2]) + '.jpg')
                image_1 = cv2.imread('./Outputs/' + str(videoUse.split('.')[-2]) + '.jpg')                

            #cv2.imshow("Cheking 000000...", image)

            if self.ui.checkBox_5.isChecked():
                
                scale_percent_0 = int(str(self.ui.lineEdit_9.text())) #25 # percent of original size
                width = int(image.shape[1] * scale_percent_0 / 100)
                height = int(image.shape[0] * scale_percent_0 / 100)
                dim = (width, height)
                image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)			
                
                #	
                width_1 = int(image_1.shape[1] * scale_percent_0 / 100)
                height_1 = int(image_1.shape[0] * scale_percent_0 / 100)
                dim_1 = (width_1, height_1)
                image_1 = cv2.resize(image_1, dim, interpolation = cv2.INTER_AREA)	
        
                cv2.imwrite('./Outputs/' + str(videoUse.split('.')[-2]) + '_Image_resized.jpg' , image)	
                #cv2.imshow("Image_resized", image)
                #cv2.waitKey(0)			

            
            if self.ui.checkBox_6.isChecked():
                
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                mask = cv2.inRange(hsv, (int(self.ui.lineEdit_2.text()), int(self.ui.lineEdit_4.text()), int(self.ui.lineEdit_6.text())), 
                (int(self.ui.lineEdit_3.text()), int(self.ui.lineEdit_5.text()), int(self.ui.lineEdit_7.text())))
                #imageObjSurface = cv2.inRange(image, (36, 0, 0), (86, 255, 255))
                
                whole_image = mask.size
                
                imask = mask>0
                green = np.zeros_like(image, np.uint8)
                green[imask] = image[imask]
                green_activ = green[imask].size
                percent_cover = (float(green_activ) / float(whole_image))*100
                print ('green pixels percentage\t\t\t:', percent_cover, '%')#, '-', percent_cover_1, '%'
                        
                cv2.imwrite('./Outputs/' + str(videoUse.split('.')[-2]) + '_GreenSurface.jpg' , green_activ)		
                        
                        
            #if not self.ui.checkBox_6.isChecked():
            if self.ui.checkBox_11.isChecked():			
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imwrite('./Outputs/' + str(videoUse.split('.')[-2]) + '_Gray_image_Filtered.jpg' , image)
                #cv2.imshow("Converted to gray", image)
                #cv2.waitKey(0)


            if self.ui.checkBox_7.isChecked():				
                #grayConverted = cv2.GaussianBlur(grayConverted, (7,7), 0) # (7, 7)- Gaussian kernels size(height, width)
                image = cv2.GaussianBlur(image, (int(self.ui.lineEdit_10.text()),int(self.ui.lineEdit_11.text())), cv2.IMREAD_UNCHANGED)
                #image = cv2.GaussianBlur(image, (7,7), 0)
                cv2.imwrite('./Outputs/' + str(videoUse.split('.')[-2]) + '_Gray_image_GBlure_Filtered.jpg' , image)
                #cv2.imshow("Gaussian Blur", image)
                #cv2.waitKey(0)

            if self.ui.checkBox_8.isChecked():
                # 15, 100 - minimum intensity gradiant, maximum intensity gradiant		
                #edgeDetect = cv2.Canny(grayConverted, 15, 100)
                
                if self.ui.checkBox_18.isChecked():
                    print ('auto canny applied!')
                    #pixelIntens = np.median(image)
                    pixelIntens = np.mean(image)
                   
                    
                    lowerSet = int(max(0, (1.0 - float(self.ui.lineEdit_37.text())) * pixelIntens))
                    upperSet = int(min(255, (1.0 + float(self.ui.lineEdit_37.text())) * pixelIntens))
                    print ('canny', 'lowerSet',lowerSet, 'upperSet', upperSet)
                    image = cv2.Canny(image, lowerSet, upperSet)
                    cv2.imwrite('./Outputs/' + str(videoUse.split('.')[-2]) + '_Canny_Filtered.jpg' , image)               
                else:
                    print (' manually setup lower and upper canny range!')
                    image = cv2.Canny(image, int(self.ui.lineEdit_12.text()), int(self.ui.lineEdit_13.text()))
                    cv2.imwrite('./Outputs/' + str(videoUse.split('.')[-2]) + '_Canny_Filtered.jpg' , image)
                
                #cv2.imshow("Canny", image)
                #cv2.waitKey(0)

            if self.ui.checkBox_9.isChecked():				
                #image = cv2.dilate(image, None, iterations=1)
                #image = cv2.dilate(image, (3,3), iterations=1)
                image = cv2.dilate(image, (int(self.ui.lineEdit_14.text()),int(self.ui.lineEdit_15.text())), iterations=int(self.ui.lineEdit_16.text()))
                cv2.imwrite('./Outputs/' + str(videoUse.split('.')[-2]) + '_Dilate_Filtered.jpg' , image)
                #cv2.imshow("Dilate", image)
                #cv2.waitKey(0)

            if self.ui.checkBox_10.isChecked():				
                #image = cv2.erode(image, None, iterations=1)
                #image = cv2.erode(image, (3,3), iterations=1)
                image = cv2.erode(image, (int(self.ui.lineEdit_17.text()),int(self.ui.lineEdit_18.text())), iterations=int(self.ui.lineEdit_19.text()))
                cv2.imwrite('./Outputs/' + str(videoUse.split('.')[-2]) + '_Erode_Filtered.jpg' , image)
                #cv2.imshow("Erode", image)
                #cv2.waitKey(0)
                
            if self.ui.checkBox_13.isChecked():				
                #image = cv2.dilate(image, None, iterations=1)
                #image = cv2.dilate(image, (3,3), iterations=1)
                image = cv2.dilate(image, (int(self.ui.lineEdit_20.text()),int(self.ui.lineEdit_21.text())), iterations=int(self.ui.lineEdit_22.text()))
                cv2.imwrite('./Outputs/' + str(videoUse.split('.')[-2]) + '_Dilate_Filtered.jpg' , image)
                #cv2.imshow("Dilate", image)
                #cv2.waitKey(0)
                    
            #
            cntours = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntours = imutils.grab_contours(cntours)	
            #(cntours, _) = contours.sort_contours(cntours)
            (cntours, _) = contours.sort_contours(cntours, method='bottom-to-top')
            #(_, cntours) = contours.sort_contours(cntours)

            #cv2.imshow("Cheking 1111111...", image)
            if str(self.ui.lineEdit_38.text()) == '':
                pixel_to_size = None
                #input('check 00')
            #if self.ui.checkBox_15.isChecked():
            else:
                pixel_to_size = float(str(self.ui.lineEdit_38.text()))
                #input('check 0')                

            def mdpt(A, B):
                return ((A[0] + B[0]) * 0.5, (A[1] + B[1]) * 0.5)

            with open('./Outputs/' + str(videoUse.split('.')[-2]) + '_width_height_allvalues.txt', 'w') as writeWidthOut, open(
            #'./Outputs/' + 'Full_auto_lenght_widht.txt', 'a') as writeFullOut, open('./Outputs/' + 'Segmented_auto_lenght_widht.txt', 'a') as writeSegOut, open(
            './Outputs/' + str(videoUse.split('.')[-2]) + '_skelet_values.txt', 'w') as writeSkelet, open(
            './Outputs/' + 'Full_skelet_values.txt', 'a') as writeSkeletFull, open(
            './BackupImages/' + 'Full_skelet_values.txt', 'a') as writeSkeletFullBackup:
           
                writeSkelet.write(                
                '{:>10}'.format('PlotNo') + '\t' 
                + '{:>10}'.format('LeafNo') + '\t' 
                + '{:>10}'.format('RankNo') + '\t' 
                + '{:>10}'.format('SegNo') + '\t'
                +  str('LnghtSumLeft') + '\t'
                +  str('LnghtSum') + '\t' 
                +  str('LnghtSumRight') + '\t'                 
                +  str('diameter')                         
                + '\n')                
            
                orig = image_1.copy()
                origskelet1 = image_1.copy()
                origskelet2 = image_1.copy()
                origskelet3 = image_1.copy()
                # loop over the contours individually			


                #if pixel_to_size is None:
                #	(cntours, _) = contours.sort_contours(cntours)
                #if pixel_to_size != None:	
                #	(cntours, _) = contours.sort_contours(cntours, method='bottom-to-top')
                mua = 0
                muaSkelet = 0
                counterWrite = 1
                
                x11 = 0
                x22 = 0
                
                x11left = 0
                x22left = 0 

                x11right = 0
                x22right = 0 

                distB11 = 0

                fullLenghtBx = 0
                fullLenghtBy = 0
                
                fullLenghtLeftBoundBx = 0
                fullLenghtLeftBoundBy = 0

                fullLenghtRightBoundBx = 0
                fullLenghtRightBoundBy = 0                
                
                sumUpLnght = 0
                sumUpLnghtLeft = 0
                sumUpLnghtRight = 0                
                
                
                
                for c in cntours:
           
                    #if cv2.contourArea(c) < 1000:
                    if cv2.contourArea(c) < (int(str(self.ui.lineEdit_23.text()))):	
                        continue
                    # compute the rotated bounding box of the contour; sho
                
                    #orig = grayImageRead.copy()
                    
                    #orig = cv2.imread('./Outputs/assebled_v.jpg')
                    
                    #orig = image.copy()
                    
                    # moved to line 488 in order to keep measured lables in one stored image!
                    #orig = image_1.copy()
                    
                    #cv2.imshow("Cheking ...", orig)
                    
                    bbox = cv2.minAreaRect(c)
                    #bbox = cv2.cv.BoxPoints(bbox) if imutils.is_cv2() else cv2.BoxPoints(bbox)
                    bbox = cv2.cv.boxPoints(bbox) if imutils.is_cv2() else cv2.boxPoints(bbox)
                    bbox = np.array(bbox, dtype="int")
                    # order the contours and draw bounding box
                    bbox = perspective.order_points(bbox)

                    mua =  mua + 1
                    #muaSkelet =  muaSkelet + 1
                    cv2.drawContours(orig, [bbox.astype("int")], -1, (0, 255, 0), 2)
                    #cv2.drawContours(orig, [bbox.astype("int")], 0, (0, 0, 255), 2)
                    
                    #scaleuUpCounter = False
                    countBox = 0
                    for (x, y) in bbox:
                        countBox = countBox + 1 
                        
                        
                        cv2.circle(orig, (int (x), int(y)), 5, (0, 0, 255), -1)
                        
                        #orig = cv2.imread('./Outputs/assebled_v.jpg')
                        
                        (tl, tr, br, bl) = bbox
                        
                        #print (tl, tr, br, bl)
                        
                        (tltrX, tltrY) = mdpt(tl, tr)
                        (blbrX, blbrY) = mdpt(bl, br)
                        
                        (tlblX, tlblY) = mdpt(tl, bl)
                        (trbrX, trbrY) = mdpt(tr, br)
                        
                        #print ('1', tltrX, tltrY, blbrX, blbrY)
                        #print ('2', tlblX, tlblY, trbrX, trbrY)
                        # draw the mdpts on the image (blue);lines between the mdpts (yellow)
                        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
                        
                        
                        #diamtart lines yellow
                        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(0, 255, 255), 2)                        
                        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(0, 255, 255), 2)

                        
                        # compute the Euclidean distances between the mdpts
                        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))		
                        #print ('dA', dA, 'dB', dB)
                            
                        if pixel_to_size is None and self.ui.checkBox_15.isChecked():
                            
                            pixel_to_size = dB / float(self.ui.lineEdit_26.text()) #2.47 # cm width 1.0cm (pixel_to_sizeA + pixel_to_sizeB)/2
                            self.ui.lineEdit_38.setText(str(round(pixel_to_size, 2)))
                            #print ('pixel to size avarage', pixel_to_size)
                            #input('check 1')
                        # use pixel_to_size ratio to compute object size
                        

                        if pixel_to_size is None:
                            #pixel_to_size = dB / args["width"]
                            #pixel_to_size = dB / 2.5 # cm width 1.0cm
                            
                            pixel_to_size = dB / float(self.ui.lineEdit_26.text()) #2.47 # cm width 1.0cm
   
                        distA = dA / pixel_to_size
                        distB = dB / pixel_to_size
                        if countBox == 4:
                            print ('countBox', countBox, 'distA', distA, 'distB', distB)
                        
                        
                        #writeWidthOut.write(str(round(distB, 3)) + '\n')
                            
                        # draw the object sizes on the image
                        #cv2.putText(orig, "{:.1f}cm".format(distA), (int(tltrX - 10), int(tltrY - 10)), cv2.FONT_HERSHEY_DUPLEX,0.55, (255, 255, 255), 2)
                        #cv2.putText(orig, "{:.1f}cm".format(distB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_DUPLEX,0.55, (255, 255, 255), 2)
                        # white font 
                        #cv2.putText(orig, "{:.1f}cm".format(distA), (int(tltrX - 10), int(tltrY - 10)), cv2.FONT_HERSHEY_DUPLEX,0.55, (255, 255, 255), 2)
                        #cv2.putText(orig, "{:.1f}cm".format(distB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_DUPLEX,0.55, (255, 255, 255), 2)
                        #black font
                        cv2.putText(orig, "{:.3f}cm".format(distA), (int(tltrX - 40), int(tltrY - 10)), cv2.FONT_HERSHEY_DUPLEX,0.55, (0, 0, 0), 2)
                        cv2.putText(orig, "{:.3f}cm".format(distB), (int(trbrX + 20), int(trbrY)), cv2.FONT_HERSHEY_DUPLEX,0.55, (0, 0, 0), 2)								


###########################
                        # testing some stuff
                        #if self.ui.checkBox_19.isChecked():
                        #muaSkelet =  muaSkelet + 1
                        
                        cv2.circle(origskelet1, (int(trbrX), int(trbrY)), 5, (0, 255, 0), -1)
                        cv2.circle(origskelet1, (int(tlblX), int(tlblY)), 5, (0, 0, 255), -1)
                        
                        
                        cv2.line(origskelet1, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 255, 0), 1)

                        #midpoint
                        #(x1, x2) = mdpt((int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)))
                        (skeletMiddleX, skeletMiddleY) = mdpt((int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)))
                        
                        #(tryScaleUpx11a, tryScaleUpx22a)  = mdpt((int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)))
                        (fullLenghtAx, fullLenghtAy)  = mdpt((int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)))                        

                        #(tltrX, tltrY) = mdpt(tl, tr)
                        #(blbrX, blbrY) = mdpt(bl, br)
                        #(tlblX, tlblY) = mdpt(tl, bl)
                        #(trbrX, trbrY) = mdpt(tr, br)
                        
                        #leftBound
                        fullLenghtLeftBoundAx = tlblX
                        fullLenghtLeftBoundAy = tlblY
                        #rightBound
                        fullLenghtRightBoundAx = trbrX
                        fullLenghtRightBoundAy = trbrY
                                                   
                        if skeletMiddleX > int(str(self.ui.lineEdit_24.text())) and x11 > int(str(self.ui.lineEdit_24.text())):

                            
  
                            #skelet middle
                            cv2.circle(origskelet1, (int(skeletMiddleX), int(skeletMiddleY)), 5, (0, 0, 0), -1)
                            cv2.line(origskelet1, (int(skeletMiddleX), int(skeletMiddleY)), (int(x11), int(x22)),(0, 0, 0), 1)                               
                            
                            skeletMiddleDist = dist.euclidean((skeletMiddleX, skeletMiddleY), (x11, x22))
                            skeletMiddleDistCm = skeletMiddleDist / pixel_to_size
                            #print ('skeletMiddleDistCm', skeletMiddleDistCm)
                            
                            #skelet left bound
                            cv2.line(origskelet1, (int(tlblX), int(tlblY)), (int(x11left), int(x22left)),(0, 0, 255), 1)
                            #skelet right bound
                            cv2.line(origskelet1, (int(trbrX), int(trbrY)), (int(x11right), int(x22right)),(0, 255, 0), 1)
                          

                            if (muaSkelet % 4) == 0:
                                                                                           
                                cv2.putText(origskelet1, "{}".format(counterWrite), (int(trbrX + 20), int(trbrY)), cv2.FONT_HERSHEY_DUPLEX,0.55, (0, 0, 0), 2)                                 

                                counterWrite = counterWrite + 1
                                
                            muaSkelet =  muaSkelet + 1        
                                                       
# #############################                        #
                        #if self.ui.checkBox_21.isChecked() and fullLenghtAx > int(str(self.ui.lineEdit_24.text())): # and x11 > int(str(self.ui.lineEdit_24.text())):
                        if fullLenghtAx > int(str(self.ui.lineEdit_24.text())):
                            #if muaSkelet == 1:
                            if counterWrite == 1:
                                fullLenghtBx = fullLenghtAx
                                fullLenghtBy = fullLenghtAy
                                
                                fullLenghtLeftBoundBx = fullLenghtLeftBoundAx
                                fullLenghtLeftBoundBy = fullLenghtLeftBoundAy

                                fullLenghtRightBoundBx = fullLenghtRightBoundAx
                                fullLenghtRightBoundBy = fullLenghtRightBoundAy    
                                
################################
                                skeletMiddleXfinal = fullLenghtAx
                                skeletMiddleYfinal = fullLenghtAy 
                                
                                fullLenghtLeftBoundBxfinal = fullLenghtLeftBoundAx
                                fullLenghtLeftBoundByfinal = fullLenghtLeftBoundAy

                                fullLenghtRightBoundBxfinal = fullLenghtRightBoundAx
                                fullLenghtRightBoundByfinal = fullLenghtRightBoundAy                                  
#################################                                
                                
                                cv2.circle(origskelet2, (int(fullLenghtAx), int(fullLenghtAy)), 10, (0, 0, 0), -1)

                                #left bound - red
                                cv2.circle(origskelet2, (int(fullLenghtLeftBoundAx), int(fullLenghtLeftBoundAy)), 5, (0, 0, 255), -1)
                                #right bound - green
                                cv2.circle(origskelet2, (int(fullLenghtRightBoundAx), int(fullLenghtRightBoundAy)), 5, (0, 255, 0), -1)

                                if countBox == 4:
                                    writeSkelet.write(
                                    '{:>10}'.format(plotNo) + '\t' 
                                    + '{:>10}'.format(leafNo) + '\t' 
                                    + '{:>10}'.format(rankNo) + '\t' 
                                    + '{:>10}'.format(str(counterWrite)) + '\t'
                                    +  str(round(sumUpLnghtLeft, 3)) + '\t'
                                    +  str(round(sumUpLnght, 3)) + '\t'
                                    +  str(round(sumUpLnghtRight, 3)) + '\t'                                    
                                    +  str(round(distB, 3))                         
                                    + '\n')
                                    
                                    writeSkeletFull.write(
                                    '{:>10}'.format(plotNo) + '\t' 
                                    + '{:>10}'.format(leafNo) + '\t' 
                                    + '{:>10}'.format(rankNo) + '\t' 
                                    + '{:>10}'.format(str(counterWrite)) + '\t'        
                                    +  str(round(sumUpLnghtLeft, 3)) + '\t'
                                    +  str(round(sumUpLnght, 3)) + '\t'
                                    +  str(round(sumUpLnghtRight, 3)) + '\t'                                    
                                    +  str(round(distB, 3))                         
                                    + '\n') 

                                    writeSkeletFullBackup.write(
                                    '{:>10}'.format(plotNo) + '\t' 
                                    + '{:>10}'.format(leafNo) + '\t' 
                                    + '{:>10}'.format(rankNo) + '\t' 
                                    + '{:>10}'.format(str(counterWrite)) + '\t'        
                                    #+  str(round(sumUpLnghtLeft, 3)) + '\t'
                                    +  str(round(sumUpLnght, 3)) + '\t'
                                    #+  str(round(sumUpLnghtRight, 3)) + '\t'                                
                                    +  str(round(distB, 3))                         
                                    + '\n')

                                
                                #give it a try
                                fullLenghtSkeletingAx = fullLenghtAx
                                fullLenghtSkeletingAy = fullLenghtAy
                            
                            #if counterWrite > 1 and (muaSkelet % 40) == 0:
                            pointObsIntesityMultiplier = int(self.ui.lineEdit_27.text())*4
                            if counterWrite > 1 and (muaSkelet % pointObsIntesityMultiplier) == 0 and countBox == 4:
                            #if counterWrite > 1 and (muaSkelet % 20) == 0:


############################# testing
                                # if self.ui.checkBox_16.isChecked():
                                    # #if fullLenghtAx < fullLenghtBx: # and fullLenghtAx > fullLenghtBx*1.3 and countBox == 1:
                                    # if (abs(fullLenghtAx - fullLenghtBx) > (fullLenghtAx*(int(self.ui.lineEdit_28.text())/100)) or
                                        # abs(fullLenghtLeftBoundAx - fullLenghtLeftBoundBx) > (fullLenghtLeftBoundAx*(int(self.ui.lineEdit_28.text())/100)) or
                                        # abs(fullLenghtRightBoundAx - fullLenghtRightBoundBx) > (fullLenghtRightBoundAx*(int(self.ui.lineEdit_28.text())/100))):
                                    # #if abs(fullLenghtAx - fullLenghtBx) > (fullLenghtAx*(int(self.ui.lineEdit_28.text())/100)): # 300:
                                        # print ('--------------------skipping distortion')
                                        # print ('--------------------skipping distortion fullLenghtAx', fullLenghtAx)
                                        # print ('--------------------skipping distortion fullLenghtBx', fullLenghtBx)
                                        # pass
                                    # else:            
                                        # cv2.circle(origskelet2, (int(fullLenghtAx), int(fullLenghtAy)), 10, (0, 0, 0), -1)
                                        # cv2.line(origskelet2, (int(fullLenghtAx), int(fullLenghtAy)), (int(fullLenghtBx), int(fullLenghtBy)),(0, 0, 0), 3)
                                        # #scaleuUpCounter = True
                                        
                                        # #left bound - red
                                        # cv2.circle(origskelet2, (int(fullLenghtLeftBoundAx), int(fullLenghtLeftBoundAy)), 10, (0, 0, 255), -1)
                                        # cv2.line(origskelet2, (int(fullLenghtLeftBoundAx), int(fullLenghtLeftBoundAy)), (int(fullLenghtLeftBoundBx), int(fullLenghtLeftBoundBy)),(0, 0, 255), 3)
                                        # #right bound - green
                                        # cv2.circle(origskelet2, (int(fullLenghtRightBoundAx), int(fullLenghtRightBoundAy)), 10, (0, 255, 0), -1)
                                        # cv2.line(origskelet2, (int(fullLenghtRightBoundAx), int(fullLenghtRightBoundAy)), (int(fullLenghtRightBoundBx), int(fullLenghtRightBoundBy)),(0, 255, 0), 3)                                  
                                        
                                        # #(me1skeletScaleUp, me2skeletScaleUp) = mdpt((int(tryScaleUpx11a), int(tryScaleUpx11b)), (int(tryScaleUpx22a), int(tryScaleUpx22b)))
                                        # #(me1skeletScaleUp, me2skeletScaleUp) = mdpt((int(fullLenghtAx), int(fullLenghtAy)), (int(fullLenghtBx), int(fullLenghtBy)))
                                        # distSkeletScaleUp = dist.euclidean((int(fullLenghtAx), int(fullLenghtAy)), (int(fullLenghtBx), int(fullLenghtBy)))
                                        # distSkeletCmScaleUp = distSkeletScaleUp / pixel_to_size
                                        
                                        # #left bound - red
                                        # distSkeletScaleUpLeftBound = dist.euclidean((int(fullLenghtLeftBoundAx), int(fullLenghtLeftBoundAy)), (int(fullLenghtLeftBoundBx), int(fullLenghtLeftBoundBy)))
                                        # distSkeletCmScaleUpLeftBound = distSkeletScaleUpLeftBound / pixel_to_size
                                        # #right bound - green
                                        # distSkeletScaleUpRightBound = dist.euclidean((int(fullLenghtRightBoundAx), int(fullLenghtRightBoundAy)), (int(fullLenghtRightBoundBx), int(fullLenghtRightBoundBy)))
                                        # distSkeletCmScaleUpRightBound = distSkeletScaleUpRightBound / pixel_to_size                                
                                        

                                        # sumUpLnght = sumUpLnght + distSkeletCmScaleUp
                                        
                                        # sumUpLnghtLeft = sumUpLnghtLeft + distSkeletCmScaleUpLeftBound
                                        # sumUpLnghtRight = sumUpLnghtRight + distSkeletCmScaleUpRightBound
                                        
                                        # cv2.putText(origskelet2, "{:.2f}cm".format(sumUpLnght), (int(trbrX + 20), int(trbrY)), cv2.FONT_HERSHEY_DUPLEX,0.55, (0, 0, 0), 2)
                                        # cv2.putText(origskelet2, "d:{:.2f}cm".format(distB), (int(trbrX + 20), int(trbrY + 20)), cv2.FONT_HERSHEY_DUPLEX,0.55, (0, 0, 0), 2)                     
                                    
                                    
                                        # writeSkelet.write(
                                        # '{:>10}'.format(plotNo) + '\t' 
                                        # + '{:>10}'.format(leafNo) + '\t' 
                                        # + '{:>10}'.format(rankNo) + '\t' 
                                        # + '{:>10}'.format(str(counterWrite)) + '\t'
                                        # +  str(round(sumUpLnghtLeft, 3)) + '\t'
                                        # +  str(round(sumUpLnght, 3)) + '\t'
                                        # +  str(round(sumUpLnghtRight, 3)) + '\t'                                
                                        # +  str(round(distB, 3)) + '\t'

                                        # +  str(round((fullLenghtAx*0.9), 3)) + '\t'
                                        # +  str(round(fullLenghtBx, 3)) + '\t'
                                        # +  str(round((fullLenghtAx*1.1), 3))
                                        
                                        # + '\n')
                                        
                                        # writeSkeletFull.write(
                                        # '{:>10}'.format(plotNo) + '\t' 
                                        # + '{:>10}'.format(leafNo) + '\t' 
                                        # + '{:>10}'.format(rankNo) + '\t' 
                                        # + '{:>10}'.format(str(counterWrite)) + '\t'        
                                        # +  str(round(sumUpLnghtLeft, 3)) + '\t'
                                        # +  str(round(sumUpLnght, 3)) + '\t'
                                        # +  str(round(sumUpLnghtRight, 3)) + '\t'                                
                                        # +  str(round(distB, 3))                         
                                        # + '\n')                                
                                    
                                    
                                    
                                        # fullLenghtBx = fullLenghtAx
                                        # fullLenghtBy = fullLenghtAy 

                                        # fullLenghtLeftBoundBx = fullLenghtLeftBoundAx
                                        # fullLenghtLeftBoundBy = fullLenghtLeftBoundAy

                                        # fullLenghtRightBoundBx = fullLenghtRightBoundAx
                                        # fullLenghtRightBoundBy = fullLenghtRightBoundAy 
###########################                                        
                                if self.ui.checkBox_17.isChecked():
                                    
                                    ################
                                    if abs(fullLenghtLeftBoundAx - fullLenghtLeftBoundBx) > (fullLenghtLeftBoundAx*(int(self.ui.lineEdit_29.text())/100)):
                                    
                                        print ('--------------------skipping distortion')
                                        print ('--------------------skipping distortion fullLenghtLeftBoundAx', fullLenghtLeftBoundAx)
                                        print ('--------------------skipping distortion fullLenghtLeftBoundBx', fullLenghtLeftBoundBx)
                                        pass
                                    else:
                                        fullLenghtLeftBoundBx = fullLenghtLeftBoundAx
                                        fullLenghtLeftBoundBy = fullLenghtLeftBoundAy                                    
                                    
                                    if abs(fullLenghtRightBoundAx - fullLenghtRightBoundBx) > (fullLenghtRightBoundAx*(int(self.ui.lineEdit_29.text())/100)):
                                    
                                        print ('--------------------skipping distortion')
                                        print ('--------------------skipping distortion fullLenghtRightBoundAx', fullLenghtRightBoundAx)
                                        print ('--------------------skipping distortion fullLenghtRightBoundBx', fullLenghtRightBoundBx)
                                        pass
                                    else:                                         
                                        fullLenghtRightBoundBx = fullLenghtRightBoundAx
                                        fullLenghtRightBoundBy = fullLenghtRightBoundAy                                    
                                    #####################
                                    
                                    
                                    if ((abs(fullLenghtLeftBoundAx - fullLenghtLeftBoundBx) < (fullLenghtLeftBoundAx*(int(self.ui.lineEdit_29.text())/100))) and 
                                        (abs(fullLenghtRightBoundAx - fullLenghtRightBoundBx) < (fullLenghtRightBoundAx*(int(self.ui.lineEdit_29.text())/100)))):
                                        
                                        cv2.putText(origskelet3, "{}".format(counterWrite), (int(trbrX + 20), int(trbrY)), cv2.FONT_HERSHEY_DUPLEX,0.55, (0, 0, 0), 2)
                                        
                                        (skeletMiddleX, skeletMiddleY) = mdpt((int(fullLenghtLeftBoundAx), int(fullLenghtLeftBoundAy)), (int(fullLenghtRightBoundAx), int(fullLenghtRightBoundAy)))
                                        
                                        cv2.circle(origskelet3, (int(skeletMiddleX), int(skeletMiddleY)), 15, (0, 0, 0), -1)
                                        cv2.circle(origskelet3, (int(fullLenghtLeftBoundAx), int(fullLenghtLeftBoundAy)), 15, (0, 0, 255), -1)
                                        cv2.circle(origskelet3, (int(fullLenghtRightBoundAx), int(fullLenghtRightBoundAy)), 15, (0, 255, 0), -1)
                                        
                                        cv2.line(origskelet3, (int(fullLenghtLeftBoundAx), int(fullLenghtLeftBoundAy)), (int(fullLenghtRightBoundAx), int(fullLenghtRightBoundAy)),(0, 0, 0), 5)
                                        
                                        #middle line length
                                        cv2.line(origskelet3, (int(skeletMiddleXfinal), int(skeletMiddleYfinal)), (int(skeletMiddleX), int(skeletMiddleY)),(0, 0, 0), 5)
                                        
                                        #left and right bound
                                        cv2.line(origskelet3, (int(fullLenghtLeftBoundBxfinal), int(fullLenghtLeftBoundByfinal)), (int(fullLenghtLeftBoundAx), int(fullLenghtLeftBoundAy)),(0, 0, 0), 5)
                                        cv2.line(origskelet3, (int(fullLenghtRightBoundBxfinal), int(fullLenghtRightBoundByfinal)), (int(fullLenghtRightBoundAx), int(fullLenghtRightBoundAy)),(0, 0, 0), 5)

                                        # calculating 
                                        distSkeletScaleUp = dist.euclidean((int(skeletMiddleXfinal), int(skeletMiddleYfinal)), (int(skeletMiddleX), int(skeletMiddleY)))
                                        distSkeletCmScaleUp = distSkeletScaleUp / pixel_to_size                                        

                                        sumUpLnght = sumUpLnght + distSkeletCmScaleUp

                                        writeSkeletFullBackup.write(
                                        '{:>10}'.format(plotNo) + '\t' 
                                        + '{:>10}'.format(leafNo) + '\t' 
                                        + '{:>10}'.format(rankNo) + '\t' 
                                        + '{:>10}'.format(str(counterWrite)) + '\t'        
                                        #+  str(round(sumUpLnghtLeft, 3)) + '\t'
                                        +  str(round(sumUpLnght, 3)) + '\t'
                                        #+  str(round(sumUpLnghtRight, 3)) + '\t'                                
                                        +  str(round(distB, 3))                         
                                        + '\n') 

                                        skeletMiddleXfinal = skeletMiddleX
                                        skeletMiddleYfinal = skeletMiddleY       

                                        fullLenghtLeftBoundBxfinal = fullLenghtLeftBoundAx
                                        fullLenghtLeftBoundByfinal = fullLenghtLeftBoundAy

                                        fullLenghtRightBoundBxfinal = fullLenghtRightBoundAx
                                        fullLenghtRightBoundByfinal = fullLenghtRightBoundAy                                         

                                        ###########
                                        fullLenghtLeftBoundBx = fullLenghtLeftBoundAx
                                        fullLenghtLeftBoundBy = fullLenghtLeftBoundAy
                                        
                                        fullLenghtRightBoundBx = fullLenghtRightBoundAx
                                        fullLenghtRightBoundBy = fullLenghtRightBoundAy
                                        ##########
##########
                                        #exceptionMiddleLeftfinal = skeletMiddleXfinal
                                        #exceptionMiddleRightfinal = skeletMiddleYfinal
##########                                        
                                    else:

                                        if abs(fullLenghtLeftBoundAx - fullLenghtLeftBoundBx) < (fullLenghtLeftBoundAx*(int(self.ui.lineEdit_29.text())/100)):    
                                            #if (counterWrite % 2) != 0:
                                            #if countBox == 4:
                                            # if abs(fullLenghtRightBoundBy - fullLenghtLeftBoundAy) < 15:
                                                # print ('fullLenghtRightBoundBy', fullLenghtRightBoundBy, 'fullLenghtLeftBoundAy', fullLenghtLeftBoundAy) 
                                                # cv2.line(origskelet3, (int(fullLenghtLeftBoundAx), int(fullLenghtLeftBoundAy)), (int(fullLenghtRightBoundBx), int(fullLenghtRightBoundBy)),(0, 0, 255), 5)
                                                # cv2.putText(origskelet3, "{}".format(counterWrite), (int(tlblX - 50), int(tlblY)), cv2.FONT_HERSHEY_DUPLEX,0.55, (0, 0, 0), 2)
                                                
                                                # (exceptionMiddleLeft, exceptionMiddleRight) = mdpt((int(fullLenghtLeftBoundAx), int(fullLenghtLeftBoundAy)), (int(fullLenghtRightBoundBx), int(fullLenghtRightBoundBy)))
                                                # cv2.circle(origskelet3, (int(exceptionMiddleLeft), int(exceptionMiddleRight)), 15, (0, 0, 255), -1)

                                            if abs(fullLenghtRightBoundBy - fullLenghtLeftBoundAy) < 15:
                                                print ('fullLenghtRightBoundBy', fullLenghtRightBoundBy, 'fullLenghtLeftBoundAy', fullLenghtLeftBoundAy) 
                                                cv2.line(origskelet3, (int(fullLenghtLeftBoundAx), int(fullLenghtLeftBoundAy)), (int(fullLenghtRightBoundBx), int(fullLenghtRightBoundBy)),(0, 0, 255), 5)
                                                cv2.putText(origskelet3, "{}".format(counterWrite), (int(tlblX - 50), int(tlblY)), cv2.FONT_HERSHEY_DUPLEX,0.55, (0, 0, 0), 2)
                                                
                                                (skeletMiddleX, skeletMiddleY) = mdpt((int(fullLenghtLeftBoundAx), int(fullLenghtLeftBoundAy)), (int(fullLenghtRightBoundBx), int(fullLenghtRightBoundBy)))
                                                cv2.circle(origskelet3, (int(skeletMiddleX), int(skeletMiddleY)), 15, (0, 0, 255), -1)
###########                                     
                                                #middle
                                                cv2.line(origskelet3, (int(skeletMiddleXfinal), int(skeletMiddleYfinal)), (int(skeletMiddleX), int(skeletMiddleY)),(0, 0, 255), 5)
                                                
                                                #left and right bound
                                                cv2.line(origskelet3, (int(fullLenghtLeftBoundBxfinal), int(fullLenghtLeftBoundByfinal)), (int(fullLenghtLeftBoundAx), int(fullLenghtLeftBoundAy)),(0, 0, 0), 5)
                                                cv2.line(origskelet3, (int(fullLenghtRightBoundBxfinal), int(fullLenghtRightBoundByfinal)), (int(fullLenghtRightBoundBx), int(fullLenghtRightBoundBy)),(0, 0, 0), 5)

                                                # calculating 
                                                distSkeletScaleUp = dist.euclidean((int(skeletMiddleXfinal), int(skeletMiddleYfinal)), (int(skeletMiddleX), int(skeletMiddleY)))
                                                distSkeletCmScaleUp = distSkeletScaleUp / pixel_to_size                                        

                                                sumUpLnght = sumUpLnght + distSkeletCmScaleUp

                                                distBexeption = dist.euclidean((int(fullLenghtLeftBoundAx), int(fullLenghtLeftBoundAy)), (int(fullLenghtRightBoundBx), int(fullLenghtRightBoundBy)))
                                                distBexeptionCalc = distBexeption / pixel_to_size
                                                
                                                writeSkeletFullBackup.write(
                                                '{:>10}'.format(plotNo) + '\t' 
                                                + '{:>10}'.format(leafNo) + '\t' 
                                                + '{:>10}'.format(rankNo) + '\t' 
                                                + '{:>10}'.format(str(counterWrite)) + '\t'        
                                                #+  str(round(sumUpLnghtLeft, 3)) + '\t'
                                                +  str(round(sumUpLnght, 3)) + '\t'
                                                #+  str(round(sumUpLnghtRight, 3)) + '\t'                                
                                                +  str(round(distBexeptionCalc, 3))                         
                                                + '\n') 

                                                fullLenghtLeftBoundBxfinal = fullLenghtLeftBoundAx
                                                fullLenghtLeftBoundByfinal = fullLenghtLeftBoundAy

                                                fullLenghtRightBoundBxfinal = fullLenghtRightBoundBx
                                                fullLenghtRightBoundByfinal = fullLenghtRightBoundBy
                                                
                                                skeletMiddleXfinal = skeletMiddleX
                                                skeletMiddleYfinal = skeletMiddleY
################                                                
                                            if self.ui.checkBox_19.isChecked() and abs(fullLenghtRightBoundBy - fullLenghtLeftBoundAy) >= 15 and abs(fullLenghtRightBoundBy - fullLenghtLeftBoundAy) < 30:
                                                print ('fullLenghtRightBoundBy', fullLenghtRightBoundBy, 'fullLenghtLeftBoundAy', fullLenghtLeftBoundAy) 
                                                cv2.line(origskelet3, (int(fullLenghtLeftBoundAx), int(fullLenghtLeftBoundAy)), (int(fullLenghtRightBoundBx), int(fullLenghtRightBoundBy)),(0, 0, 255), 5)
                                                cv2.putText(origskelet3, "{}".format(counterWrite), (int(tlblX - 50), int(tlblY)), cv2.FONT_HERSHEY_DUPLEX,0.55, (0, 0, 0), 2)

                                                (exceptionMiddleLeft1, exceptionMiddleRight1) = mdpt((int(fullLenghtLeftBoundAx), int(fullLenghtLeftBoundAy)), (int(fullLenghtRightBoundBx), int(fullLenghtRightBoundBy)))
                                                cv2.circle(origskelet3, (int(exceptionMiddleLeft1), int(exceptionMiddleRight1)), 15, (0, 0, 255), -1)

############################# testing

                                #if not self.ui.checkBox_16.isChecked() and not self.ui.checkBox_17.isChecked():
                                if not self.ui.checkBox_17.isChecked():
                                    cv2.circle(origskelet2, (int(fullLenghtAx), int(fullLenghtAy)), 10, (0, 0, 0), -1)
                                    cv2.line(origskelet2, (int(fullLenghtAx), int(fullLenghtAy)), (int(fullLenghtBx), int(fullLenghtBy)),(0, 0, 0), 3)
                                    #scaleuUpCounter = True
                                    
                                    #left bound - red
                                    cv2.circle(origskelet2, (int(fullLenghtLeftBoundAx), int(fullLenghtLeftBoundAy)), 10, (0, 0, 255), -1)
                                    cv2.line(origskelet2, (int(fullLenghtLeftBoundAx), int(fullLenghtLeftBoundAy)), (int(fullLenghtLeftBoundBx), int(fullLenghtLeftBoundBy)),(0, 0, 255), 3)
                                    #right bound - green
                                    cv2.circle(origskelet2, (int(fullLenghtRightBoundAx), int(fullLenghtRightBoundAy)), 10, (0, 255, 0), -1)
                                    cv2.line(origskelet2, (int(fullLenghtRightBoundAx), int(fullLenghtRightBoundAy)), (int(fullLenghtRightBoundBx), int(fullLenghtRightBoundBy)),(0, 255, 0), 3)                                  
                                    
                                    #(me1skeletScaleUp, me2skeletScaleUp) = mdpt((int(tryScaleUpx11a), int(tryScaleUpx11b)), (int(tryScaleUpx22a), int(tryScaleUpx22b)))
                                    #(me1skeletScaleUp, me2skeletScaleUp) = mdpt((int(fullLenghtAx), int(fullLenghtAy)), (int(fullLenghtBx), int(fullLenghtBy)))
                                    distSkeletScaleUp = dist.euclidean((int(fullLenghtAx), int(fullLenghtAy)), (int(fullLenghtBx), int(fullLenghtBy)))
                                    distSkeletCmScaleUp = distSkeletScaleUp / pixel_to_size
                                    
                                    #left bound - red
                                    distSkeletScaleUpLeftBound = dist.euclidean((int(fullLenghtLeftBoundAx), int(fullLenghtLeftBoundAy)), (int(fullLenghtLeftBoundBx), int(fullLenghtLeftBoundBy)))
                                    distSkeletCmScaleUpLeftBound = distSkeletScaleUpLeftBound / pixel_to_size
                                    #right bound - green
                                    distSkeletScaleUpRightBound = dist.euclidean((int(fullLenghtRightBoundAx), int(fullLenghtRightBoundAy)), (int(fullLenghtRightBoundBx), int(fullLenghtRightBoundBy)))
                                    distSkeletCmScaleUpRightBound = distSkeletScaleUpRightBound / pixel_to_size                                
                                    

                                    sumUpLnght = sumUpLnght + distSkeletCmScaleUp
                                    
                                    sumUpLnghtLeft = sumUpLnghtLeft + distSkeletCmScaleUpLeftBound
                                    sumUpLnghtRight = sumUpLnghtRight + distSkeletCmScaleUpRightBound
                                    
                                    cv2.putText(origskelet2, "{:.2f}cm".format(sumUpLnght), (int(trbrX + 20), int(trbrY)), cv2.FONT_HERSHEY_DUPLEX,0.55, (0, 0, 0), 2)
                                    cv2.putText(origskelet2, "d:{:.2f}cm".format(distB), (int(trbrX + 20), int(trbrY + 20)), cv2.FONT_HERSHEY_DUPLEX,0.55, (0, 0, 0), 2)                     
                                
                                
                                    writeSkelet.write(
                                    '{:>10}'.format(plotNo) + '\t' 
                                    + '{:>10}'.format(leafNo) + '\t' 
                                    + '{:>10}'.format(rankNo) + '\t' 
                                    + '{:>10}'.format(str(counterWrite)) + '\t'
                                    +  str(round(sumUpLnghtLeft, 3)) + '\t'
                                    +  str(round(sumUpLnght, 3)) + '\t'
                                    +  str(round(sumUpLnghtRight, 3)) + '\t'                                
                                    +  str(round(distB, 3)) + '\t'

                                    +  str(round((fullLenghtAx*0.9), 3)) + '\t'
                                    +  str(round(fullLenghtBx, 3)) + '\t'
                                    +  str(round((fullLenghtAx*1.1), 3))
                                    
                                    + '\n')
                                    
                                    writeSkeletFull.write(
                                    '{:>10}'.format(plotNo) + '\t' 
                                    + '{:>10}'.format(leafNo) + '\t' 
                                    + '{:>10}'.format(rankNo) + '\t' 
                                    + '{:>10}'.format(str(counterWrite)) + '\t'        
                                    +  str(round(sumUpLnghtLeft, 3)) + '\t'
                                    +  str(round(sumUpLnght, 3)) + '\t'
                                    +  str(round(sumUpLnghtRight, 3)) + '\t'                                
                                    +  str(round(distB, 3))                         
                                    + '\n')                                
                                
                                
                                
                                    fullLenghtBx = fullLenghtAx
                                    fullLenghtBy = fullLenghtAy 

                                    fullLenghtLeftBoundBx = fullLenghtLeftBoundAx
                                    fullLenghtLeftBoundBy = fullLenghtLeftBoundAy

                                    fullLenghtRightBoundBx = fullLenghtRightBoundAx
                                    fullLenghtRightBoundBy = fullLenghtRightBoundAy 

                                
# ##################################
    
    
                    if skeletMiddleX > int(str(self.ui.lineEdit_24.text())): # and x11 > int(str(self.ui.lineEdit_24.text())):            
                    #if x1 > int(str(self.ui.lineEdit_24.text())) and x11 > int(str(self.ui.lineEdit_24.text())):    
                        x11 = int(skeletMiddleX)
                        x22 = int(skeletMiddleY)

                        x11left = int(tlblX)
                        x22left = int(tlblY)

                        x11right = int(trbrX)
                        x22right = int(trbrY)                    
                    
                    #    
                    writeWidthOut.write(str(mua) + '\t\t' +  str(round(distA, 3)) + '\t\t' +  str(round(distB, 3)) + '\n')

########## Last point
                #pixel_to_size = float(str(self.ui.lineEdit_38.text()))        
                # middle last point 
                cv2.circle(origskelet2, (int(x11), int(x22)), 15, (0, 0, 0), -1)
                cv2.line(origskelet2, (int(fullLenghtBx), int(fullLenghtBy)), (int(x11), int(x22)),(0, 0, 0), 3)
                
                # left bound last point - red
                cv2.circle(origskelet2, (int(x11left), int(x22left)), 15, (0, 0, 255), -1)
                cv2.line(origskelet2, (int(fullLenghtLeftBoundBx), int(fullLenghtLeftBoundBy)), (int(x11left), int(x22left)),(0, 0, 0), 3)
                # right bound last point - green
                cv2.circle(origskelet2, (int(x11right), int(x22right)), 15, (0, 255, 0), -1)
                cv2.line(origskelet2, (int(fullLenghtRightBoundBx), int(fullLenghtRightBoundBy)), (int(x11right), int(x22right)),(0, 0, 0), 3)                 

                #middle
                distSkeletScaleUp = dist.euclidean((int(fullLenghtBx), int(fullLenghtBy)), (int(x11), int(x22)))
                distSkeletCmScaleUp = distSkeletScaleUp / pixel_to_size
                                
                #left bound - red
                distSkeletScaleUpLeftBound = dist.euclidean((int(fullLenghtLeftBoundBx), int(fullLenghtLeftBoundBy)), (int(x11left), int(x22left)))
                distSkeletCmScaleUpLeftBound = distSkeletScaleUpLeftBound / pixel_to_size
                        
                        
                #right bound - green
                distSkeletScaleUpRightBound = dist.euclidean((int(fullLenghtRightBoundBx), int(fullLenghtRightBoundBy)), (int(x11right), int(x22right)))
                distSkeletCmScaleUpRightBound = distSkeletScaleUpRightBound / pixel_to_size                                
                

                #sumUpLnght = sumUpLnght + distSkeletCmScaleUp
                sumUpLnght = distSkeletCmScaleUp
                
                #sumUpLnghtLeft = sumUpLnghtLeft + distSkeletCmScaleUpLeftBound
                #sumUpLnghtRight = sumUpLnghtRight + distSkeletCmScaleUpRightBound
                sumUpLnghtLeft = distSkeletCmScaleUpLeftBound
                sumUpLnghtRight = distSkeletCmScaleUpRightBound                
                

                dB = dist.euclidean((x11left, x22left), (x11right, x22right))
                distB = dB / pixel_to_size

                writeSkelet.write(
                '{:>10}'.format(plotNo) + '\t' 
                + '{:>10}'.format(leafNo) + '\t' 
                + '{:>10}'.format(rankNo) + '\t' 
                + '{:>10}'.format(str(counterWrite)) + '\t'
                +  str(round(sumUpLnghtLeft, 3)) + '\t'
                +  str(round(sumUpLnght, 3)) + '\t'
                +  str(round(sumUpLnghtRight, 3)) + '\t'                                
                +  str(round(distB, 3))                         
                + '\n')
                
                writeSkeletFull.write(
                '{:>10}'.format(plotNo) + '\t' 
                + '{:>10}'.format(leafNo) + '\t' 
                + '{:>10}'.format(rankNo) + '\t' 
                + '{:>10}'.format(str(counterWrite)) + '\t'        
                +  str(round(sumUpLnghtLeft, 3)) + '\t'
                +  str(round(sumUpLnght, 3)) + '\t'
                +  str(round(sumUpLnghtRight, 3)) + '\t'                                
                +  str(round(distB, 3))                         
                + '\n')             

                cv2.putText(origskelet2, "{:.2f}cm".format(sumUpLnght), (int(x11right + 20), int(x22right)), cv2.FONT_HERSHEY_DUPLEX,0.55, (0, 0, 0), 2)
                cv2.putText(origskelet2, "d:{:.2f}cm".format(distB), (int(x11right + 20), int(x22right + 20)), cv2.FONT_HERSHEY_DUPLEX,0.55, (0, 0, 0), 2)   

                # connect first and last point directly here
                fullLenghtSkeletingBx = x11
                fullLenghtSkeletingBy = x22

                cv2.circle(origskelet2, (int(fullLenghtSkeletingAx), int(fullLenghtSkeletingAy)), 15, (0, 0, 0), -1)
                cv2.circle(origskelet2, (int(fullLenghtSkeletingBx), int(fullLenghtSkeletingBy)), 15, (0, 0, 0), -1)
                cv2.line(origskelet2, (int(fullLenghtSkeletingAx), int(fullLenghtSkeletingAy)), (int(fullLenghtSkeletingBx), int(fullLenghtSkeletingBy)),(0, 0, 0), 3)
                
########## Last point

                cv2.imwrite('./Outputs/' + str(videoUse.split('.')[-2]) + '_Image.jpg' , orig)
                cv2.imwrite('./Outputs/' + str(videoUse.split('.')[-2]) + '_SegmentedLenght_Image_skelet.jpg' , origskelet1)
                cv2.imwrite('./Outputs/' + str(videoUse.split('.')[-2]) + '_SegmentedLenght_Image_skelet_Processed.jpg' , origskelet2)
                cv2.imwrite('./Outputs/' + str(videoUse.split('.')[-2]) + '_SegmentedLenght_Image_skelet_scaleup_Processed_additionally.jpg' , origskelet3)
                
                cv2.imwrite('./BackupImages/' + str(videoUse.split('.')[-2]) + '_Image.jpg' , orig)
                cv2.imwrite('./BackupImages/' + str(videoUse.split('.')[-2]) + '_SegmentedLenght_Image_skelet.jpg' , origskelet1)
                cv2.imwrite('./BackupImages/' + str(videoUse.split('.')[-2]) + '_SegmentedLenght_Image_skelet_Processed.jpg' , origskelet2)
                cv2.imwrite('./BackupImages/' + str(videoUse.split('.')[-2]) + '_SegmentedLenght_Image_skelet_Processed_additionally.jpg' , origskelet3)



                
                #
                #cv2.circle(origskelet2, (int(fullLenghtAx), int(fullLenghtAy)), 10, (0, 0, 0), -1)
                #cv2.line(origskelet2, (int(fullLenghtAx), int(fullLenghtAy)), (int(fullLenghtBx), int(fullLenghtBy)),(0, 0, 0), 3)

                    
        print ('Finished ....')
    

    def full_auto_lenght_width(self):
    
        print ('full auto execute!')
      
        QApplication.processEvents()                
        self.convert_mp4_to_jpg()
        QApplication.processEvents()        
        self.find_edges()
        
        
    def closeEvent(self, event):

        print ('EXIT all running threads...')
        sys.exit(app.exec_())
        #app.exit()        
        
        
if __name__ == '__main__':
    
    #app = QtWidgets.QApplication(sys.argv)
    #for qt5
    app = QApplication(sys.argv)
    Interface = WindowInt()
    Interface.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.WindowMaximizeButtonHint)	
    Interface.show()
    sys.exit(app.exec_())			        