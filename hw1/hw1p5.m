close all;

A=double(rgb2gray(imread('Data/harvey-saturday-goes7am.jpg')));
[m,n] = size(A);

rank_lst = [2 10 40];

errors = zeros(length(rank_lst),1);
figure;

for k=1:length(rank_lst)
    [U,S,V] = svd(A);
	
    rank_k = rank_lst(k);
	A_rec = U(:,1:rank_k) * S(1:rank_k,1:rank_k) * V(:,1:rank_k)';
 
    subplot(2,2,k);
    hold on;
    axis equal;
    imagesc(flipud(A_rec));
    title(['rank-' num2str(rank_k) ' approximation']);
    set(gca,'YTick',[])
    set(gca,'XTick',[])
%     colormap(gray);
    errors(k) = norm(A-A_rec,'fro')/norm(A,'fro');
    hold off;
end

subplot(2,2,4);
imagesc(A);
title('Original Image');

% figure;
% plot(rank_lst,errors);
% xlabel('rank') 
% ylabel('norm_{F} error')
% title('Errors of different k');
