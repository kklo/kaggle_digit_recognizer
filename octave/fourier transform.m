
raw_file = csvread("../data/train.csv");

# Too many data, let try the first 50 rows
data = raw_file(1:50, :);

idx = (data(:, 1)==0);
rows_0 = data(idx,:);
idx = (data(:, 1)==1);
rows_1 = data(idx,:);
idx = (data(:, 1)==2);
rows_2 = data(idx,:);
idx = (data(:, 1)==3);
rows_3 = data(idx,:);
idx = (data(:, 1)==4);
rows_4 = data(idx,:);
idx = (data(:, 1)==5);
rows_5 = data(idx,:);
idx = (data(:, 1)==6);
rows_6 = data(idx,:);
idx = (data(:, 1)==7);
rows_7 = data(idx,:);
idx = (data(:, 1)==8);
rows_8 = data(idx,:);
idx = (data(:, 1)==9);
rows_9 = data(idx,:);

#pick one to have a look
r9 = rows_9(2,:);
m = reshape(r9(2:end), 28, 28)';
figure(1); imshow(m, [0 250]); title("Original");
m_fft = fftshift(fft2(m)); 
figure(2); imshow(abs(m_fft), []); title("Magnitude");
figure(3); imshow(log(1+abs(m_fft)),[]); title("Log magnitude"); #reduce contract
