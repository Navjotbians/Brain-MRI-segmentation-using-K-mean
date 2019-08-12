clear all
close all
clc

%//////////// loading of sample in matlab //////
no= "name or number of the image";

image=imread(strcat('Path of test image',num2str(no),'.bmp'));
gt=imread(strcat('Path to the ground truth image',num2str(no),'.bmp'));
image=image(:,:,1);
figure(1)
imshow(image)
title('orignal image')
drawnow

%////////// noise and skull removing /////////////////////


image_1=image>50;           % >50 then make those pixels white
figure(2)
imshow(image_1)
title('thresholding with 50')
drawnow

image_2=bwareaopen(image_1,100);   % objects with less then 100 pixels are removed or noise removed
figure(3)
imshow(image_2)
title('morphological noise removing')
drawnow

se=strel('disk',5);
image_3=imerode(image_2,se);
figure(4)
imshow(image_3)
title('after errosion operation')
drawnow

[lab num]=bwlabel(image_3);   % labeling object, num will tell how many objects are their and lab is matrix or labeled image
figure(5)
imshow(lab)
title('labeled bw image')
drawnow               
                          %//// loop starts from 1 to num,it will
                                % calculate the area of each labeled object
                                % and matrix is created

for i=1:num
    image_4=lab==i;
    p=sum(sum(image_4));
    areas(i,:)=[i p];
end
areas=sortrows(areas,2);        % areas shorted in accending orders so that we can get largest value of area in end row and 2 is because area is in second coloum
image_aoi=lab==areas(end,1);     % our aoi is end row first coloum of area matrix ////
figure(6)
imshow(image_aoi)
title('exracted aoi')            % AOI has largest area
drawnow

se=strel('disk',14);                   % to nullify errode, dilation is done
image_aoi=imdilate(image_aoi,se);
figure(7)
imshow(image_aoi)
title('AOI after dilation operation')
drawnow

image_aoi=imfill(image_aoi,'holes');         % to fill inside portion of AOI
figure(8)
imshow(image_aoi)
title('AOI with filling morphological operation')
drawnow

se=strel('disk',11);
image_aoi=imerode(image_aoi,se);
mask=image_aoi;

image_aoi=image .* uint8(image_aoi);       % orignal image multiplied with AOI or MASK and logical image is converted into uint8 for this
figure(9)
imshow(image_aoi)
title('Extracted gray scale AOI')            % Brain tissue obtained and skull is removed
drawnow



% ////////////// segmentation ///////////////////////////

size_img=size(image_aoi);

img = image_aoi;
img_1=reshape(img,size_img(1)*size_img(2),1);                    % reshape image in one coloum and sizeImg1*sizeImg2 rows(1st element*2nd element=65536)////

img_1 = double(img_1);                                               % changing from uint8 to double format         
q_colors=5;                                                         % dividing image into five groups or intensities
[idx, C] = kmeans(img_1,q_colors, 'EmptyAction', 'singleton');      % c assign intensities to the 5 groups and idx contain pixel value for grouping purpose according to c
dummy = round(C);                                                 % decimal values of c are round off in integer values


temp = reshape(idx,size_img(1),size_img(2));               %  reshape idx into row coloum matrix /////// next loop for % creating out image using values of idx.Reading rows and coloums to assign intensities accordingly and creating clusters
                          
for i = 1:size_img(1)
    for j = 1:size_img(2)
        out_img(i,j) = dummy(temp(i,j));            %  saving values of dummy in out img according to temp
    end
end
out_img=uint8(out_img);                         % changing from double to uint8

figure(10)
imshow(out_img)
title('image with k means clustring')
drawnow

dummy=sortrows(dummy,1);                        % shorting values of dummy in asscending order so that we can get same intensities every time in respective clusters

img_e=edge(out_img,'sobel');                          % edge detection
figure(11)
imshow(img_e)
drawnow

m=~logical(mod(image,2));               % closing operation using frequecy domain or internal edge merging
m1=edge(m,'sobel');
img_e1=img_e & m;
img_e2=img_e1 | m1;

se=strel('disk',1);
img_e2=imdilate(img_e2,se);
img_e2=~bwareaopen(img_e2,50);      % nagete img_e2

img_e2=img_e2 & mask;                % multiply img_e2 with AOI mask to get segments
img_e2=bwareaopen(img_e2,50);
figure(12)
imshow(img_e2)
title('segmented objects')
drawnow

%/////////// object labelling and feature extraction ////////////

[lab num]=bwlabel(img_e2);      % label segmented image
figure(13)
imshow(lab)
drawnow

[m n]=size(image_aoi);     % size of image_aoi
s=0;                          % sum
p=0;                           % pixels     /////// for loop initialized for rows and coloums 
for i=1:m
    for j=1:n
        if(image_aoi(i,j)>0)           % if pixel has value greater then 0 then add that pixel into sum(we have to add values of white portion only and 0 is black colour value)
            s=s + double(image_aoi(i,j));    % format changed into double because we are adding values of pixels and it will go higher then 256
            p=p+1;                        % add pixels or increament
        end
    end
end
mean_int=s/p;          % mean intensity of image_aoi


for i=1:num
    img=lab==i;        % object wise area is calculated
    [r c]=find(img==1);
    area=length(r);
    
    L=max(r)-min(r);     %length of object
    B=max(c)-min(c);      % breadth of object
    C=abs(L-B);           % absolute differenc b/w L & B
    
    int=[];                      % empty matrix initiated with name intensity
    for j=1:length(r)              % object wise values of r & c are readed and stored into int matrix
        int=[int;image(r(j),c(j))];
    end
    std=abs(mean(int) - mean_int);     %standard daviation calculated (mean intensity of image_aoi- mean intensity of perticular object)
    
    x=mean(r);                  % mean of perticular objects row
    y=mean(c);                  % mean of particular objects coloum
    d=sqrt( ((m/2)-x).^2 + ((n/2)-y).^2);    % distance calculated
    
    fvt(:,i)=[i;area;C;std;d];     % feature vector table matrix created
end

%//////////// removing NAOI ///////////////////////////////////////// 
fvt=fvt';
fvt=sortrows(fvt,2);
fvt=fvt(1:end-1,:);   % object with largest area can't be tumor so we eliminated that object here. Set of rules
fvt=fvt';

%///////////// tumour detectin using weighted sum method ////////////

w1=0.5;      % area (high)
w2=.2;       % L-B  (low)
w3=0.2;      % std   (high)
w4=0.1;      % distance (high)

for i=1:size(fvt,2)
    a1=((fvt(2,i) * w1)/max(fvt(2,:)));
    a2=w2-((fvt(3,i) * w2)/max(fvt(3,:)));
    a3=((fvt(4,i) * w3)/max(fvt(4,:)));
    a4=((fvt(5,i) * w4)/max(fvt(5,:)));
    
    a=a1+a2+a3+a4;
    R(i,:)=[fvt(1,i) a1 a2 a3 a4 a];
end
R=sortrows(R,6);      % rank matrix is shorted

img_t=lab==R(end,1);         % higher rank object is assigned to img_t that is detected tumor
se=strel('disk',1);
img_t=imdilate(img_t,se);
figure(14)
imshow(img_t)
title('detected tumor')
drawnow


figure(15)
subplot(1,2,1)
imshow(gt)
title('groud truth')
subplot(1,2,2)
imshow(img_t)
title('proposed tumor detected');    % gt and detected tumor displayed
drawnow

%/////////// concluding errors ///////////////////////////

flase_alarm=~gt & img_t;   % false alarm detected
figure(16)
imshow(flase_alarm);
title('flase alarm tumour portion')
drawnow
 
miss_alarm=gt & ~img_t;      % miss alarm detected
figure(17)
imshow(miss_alarm);
title('miss alarm tumour portion')
drawnow

flase_alarm_val=sum(sum(double(flase_alarm)))    % false alarm
miss_alarm_val=sum(sum(double(miss_alarm)))      % miss alarm

overall_error=flase_alarm_val + miss_alarm_val     % overall error
[m n]=size(image);
acc=(1-((overall_error)/(m*n))) * 100        % accuracy





           
    
    







