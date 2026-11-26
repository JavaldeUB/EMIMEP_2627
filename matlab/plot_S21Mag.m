function h = plot_S21Mag(Xp,yp, freqAxis)
    uniqueNumbers = sort(unique(yp));
    colors = jet(length(uniqueNumbers));
    h = figure;
    legend_handles = gobjects(length(uniqueNumbers), 1);
    hold on;
    for i = 1:length(uniqueNumbers)
        label = uniqueNumbers(i);
        rows = Xp(yp == label, :);

        for j = 1:size(rows, 1)
            %[val, idx]=min(20.*log10(abs(rows(j,:)./rowsb)));
            hix = plot(freqAxis,rows(j, :), 'Color', colors(i, :));        
            %text(freqAxis(idx), val, num2str(j), ...
            %    'HorizontalAlignment', 'left', ...
            %    'VerticalAlignment', 'middle');
            if j == 1
                legend_handles(i) = hix;
            end
        end
    end
    hold off;
    xlabel("Freq (GHz)");
    ylabel("S_{21} (dB)");
    legend(legend_handles, arrayfun(@num2str, uniqueNumbers, 'UniformOutput', false), 'Location', 'best');
    xlim([min(freqAxis) max(freqAxis)])
end