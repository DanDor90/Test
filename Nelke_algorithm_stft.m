clear all
close all
clc


%% Load samples of pure wind and clean speech and mix them together with a fixed SNR

[n_1 Fs] = audioread('mic1_2cm.wav');                  % first microphone wind noise
[n_2 Fs] = audioread('mic2_2cm.wav');                  % second microphone wind noise

[x Fs] = audioread('male_english_resampled_16k.wav');  % mono speech signal

%{
x = [x x];                                             % stereo speech signal
n = [n_1 n_2];                                         % stereo wind noise signal

n_len = length(n(:,1));
s_len = length(x(:,1));

snr = 0;                                              % fixed SNR
dbs = -20;                                             % speech dB
dbn = dbs-snr;                                         % wind dB

if s_len>n_len                                         % align the wind noise duration with speech
diff = s_len/n_len;
rep = 5*round(diff+1);
noise = repmat(n,[rep,1]);
else 
    noise = n;
end

x_1 =  x(:,1)/max(abs(x(:,1)));                        % fix the 1st channel of speech to the dB value
    rms_sound_dB1 = norm(x_1)/sqrt(s_len);
    ratio1 = (10^(dbs/20))/rms_sound_dB1;
    signal1 = ratio1*x_1;
    
x_2 =  x(:,2)/max(abs(x(:,2)));                        % fix the 2nd channel of speech to the dB value
    rms_sound_dB2 = norm(x_2)/sqrt(s_len);
    ratio2 = (10^(dbs/20))/rms_sound_dB1;
    signal2 = ratio2*x_2;
    
noise1 =  noise(:,1)/max(abs(noise(:,1)));             % fix the 1st channel of wind noise to the dB value
    rms_sound_dBn1 = norm(noise(:,1))/sqrt(n_len);
    ratio_n1 = (10^(dbn/20))/rms_sound_dBn1;
    noise1 = ratio_n1*noise1;
    
noise2 =  noise(:,2)/max(abs(noise(:,2)));             % fix the 2nd channel of wind noise to the dB value
    rms_sound_dBn2 = norm(noise2)/sqrt(n_len);
    ratio_n2 = (10^(dbn/20))/rms_sound_dBn2;
    noise2 = ratio_n2*noise2;
    
mix1 = signal1+noise1(1:length(signal1));              % mix of wind noise and speech signal for both channels
mix2 = signal2+noise2(1:length(signal2));

mix = [mix1 mix2];                                     % stereo file of mixed signals
%}

z1 = v_addnoise(x,Fs,-5,'dbkx',n_1,Fs);
z2 = v_addnoise(x,Fs,-5,'dbkx',n_2,Fs);


x1 = z1(1:length(x),1)+z1(1:length(x),2);              % First microphone
x2 = z2(1:length(x),1)+z2(1:length(x),2);              % Second microphone
n1 = z1(1:length(x),2);                                % Real wind (for evaluation purposes)

%% Init parameters for processing

K = 512;                                               % FFT resolution

X = stft(x,0.02*Fs,0.005*Fs,512,Fs);               % STFT of the reference signal
X1 = stft(x1,0.02*Fs,0.005*Fs,512,Fs);             % STFT of the first channel
X2 = stft(x2,0.02*Fs,0.005*Fs,512,Fs);             % STFT of the second channel
N1 = stft(n1,0.02*Fs,0.005*Fs,512,Fs);             % STFT of the pure wind signal


% Parameters for the computation of the overestimation factor and the minimum gain

G_n = 10^(-6);
G_s = 10^(-0.5);
gamma_s = 1;
gamma_n = 5;
sigma_m = 0.2;
sigma_M = 0.5;

% Init of the quantities used in the processing

Pxx1 = zeros(K/2+1,1);                        % Estimated Auto-PSDs
Pxx2 = zeros(K/2+1,1);

Pxy = zeros(K/2+1,1);                         % Estimated Cross-PSD

Pnn = zeros(K/2+1,1);                         % Estimated Wind noise PSD

Pnn_th = zeros(K/2+1,1);                      % Theoretical Wind noise PSD (Dorbecker)

Pnn_real = zeros(K/2+1,1);                    % Real wind auto-PSD   

Coh = zeros(K/2+1);                           % Complex Coherence

G = zeros(K/2+1,length(X1(1,:)));             % Spectral Gain

S = zeros(K/2+1,length(X1(1,:)));             % Enhanced signal STFT

alpha = 0.8;                                  % Initial smoothing factor

alpha_v = zeros(length(X1(1,:)),1);           % Array of smotthing factor values

%% Processing for every frame

for i=1:length(X1(1,:))                                 % for every window
    
    X1_2 = abs(X1(:,i)).^2;                             % Auto-Power Spectrum of signal 1
    
    X2_2 = abs(X2(:,i)).^2;                             % Auto-Power Spectrum of signal 2
    
    X12 = X1(:,i).*conj(X2(:,i));                       % Cross-Power Spectrum of signal 1-2
    
    Pxx1 = (alpha).*Pxx1 + (1-alpha).*X1_2;             % Estimated Auto-PSD of signal 1
    
    Pxx2 = (alpha).*Pxx2 + (1-alpha).*X2_2;             % Estimated Auto-PSD of signal 2
    
    Pxy = (alpha).*Pxy + (1-alpha).*X12;                % Estimated Cross-PSD of signals 1-2
    
    Pnn_real = abs(N1(:,i)).^2;                         % Real PSD of the wind noise (for evaluation)    
    
    %% Detection phase
    
    Coh = Pxy./sqrt(Pxx1.*Pxx2);                        % Compute the complex coherence
    
    C_ph = angle(Coh);                                  % Compute the phase of the complex coherence
    
    sigma_l(i) = min((3/pi^2)*1/126*sum((C_ph(2:128)-...
                 mean(C_ph(2:128))).^2),1);             % Compute the normalized variance (values between 0 and 1) for low frequency
    
    sigma_h(i) = min((3/pi^2)*sum((C_ph(128:256)-...
                 mean(C_ph(128:256))).^2/126),1);       % Compute the normalized variance (values between 0 and 1) for high frequency
    
    %% Estimation Phase
    
    Pnn_th = sqrt(Pxx1.*Pxx2) - abs(Pxy);               % Compute the theoretical noise PSD (Dorbecker)
    
    Pnn = (1-sigma_l(i)).*Pnn_th + (sigma_l(i)).*X1_2;  % Weight the th. PSD with the actual input power spectrum
    
    %% Update of the smoothing factor
    
    alpha = 1 -(sigma_l(i)/2);                          % Linear mapping between the variance of the phase of the coherence and the smoothing factor
    
    
    if alpha < 0.5                                      % Truncation for max-min values of smoothing parameter
     alpha = 0.5;
    elseif alpha >0.98
     alpha = 0.98;
    end
    
    alpha_v(i) = alpha;                                 % Store the smoothing value in the current frame
    
    
    
    %% Reduction Phase
    
    G_min = (1/(sigma_M-sigma_m))*(G_n*(sigma_h(i)-...  % Compute the minimum spectral gain
             sigma_m)+G_s*(sigma_M-sigma_h(i)));
    
    gamma = (1/(sigma_M-sigma_m))*(gamma_n*...          % Compute the overestimation factor
    (sigma_h(i)-sigma_m)+gamma_s*(sigma_M-sigma_h(i)));

    
    
    G(:,i) = max((1-(Pnn./X1_2)),...
             0.1);                                    % Compute the spectral gains
    
    S(:,i) = G(:,i).*X1(:,i);                           % Filtering of the noisy input signal 1 = enhanced spectrum
    
    
end

s = istft(S, 0.02*Fs, 0.005*Fs, K, Fs);              % iSTFT of the enhanced spectrum = enhanced time signal 

x_ev = istft(X, 0.02*Fs, 0.005*Fs, K, Fs);           % iSTFT of the reference signal for evaluation

x1_i = istft(X1, 0.02*Fs, 0.005*Fs, K,Fs);           % iSTFT of the noisy input signal for evaluation

%% Plot noisy input and enhanced signal

figure(1)
subplot(2,1,1)
spectrogram(x1(1:10*Fs),hanning(320),160,512,Fs,'yaxis'),title('Corrupted signal');
subplot(2,1,2)
spectrogram(s(1:10*Fs),hanning(320),160,512,Fs,'yaxis'),title('Enhanced signal');

%% Plot the variance over the time signal to see how the code detect wind noise

figure(2)
subplot(3,1,1)
plot((0:(length(x1)-1))/Fs,x1),ylabel('Amplitude'), xlabel('Time index'), axis tight, title('Corrupted input signal')
subplot(3,1,2)
plot(1:length(X1(1,:)), sigma_l),ylabel('Variance \sigma^2'), xlabel('Frame index'), title('Variance of the phase of the Coherence for every frame'),axis tight
subplot(3,1,3)
plot(1:length(X1(1,:)), alpha_v, '-r'),ylabel('Smoothing factor \alpha'), xlabel('Frame index'), title('Smoothing factor for every frame'), axis tight

%% Performance evaluation

pesq_s = pesq(x_ev,s,Fs);

pesq_n = pesq(x_ev,x1_i,Fs);

pesq_o = pesq(x_ev,x_ev,Fs);

p = pesq_s - pesq_n;

stoi_s = stoi(x_ev,s,Fs);

stoi_n = stoi(x_ev,x1_i,Fs);

st = stoi_s -stoi_n;

md_s = mfcc_distance(x_ev, s, Fs);

md_n = mfcc_distance(x_ev, x1_i, Fs);

m = md_s - md_n;

d_s = fwsegsnr(s, x_ev, Fs);

d_n = fwsegsnr(x1, x_ev, Fs);

d = d_s - d_n;