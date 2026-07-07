function metrics = chaotic(csv_path, out_json_path)
close all;
if nargin < 1 || strlength(string(csv_path)) == 0

    error('csv_path is required');

end

if nargin < 2

    out_json_path = '';

end



tic
filename='simulation_results';
cadence_data=readmatrix(csv_path);



do_plots = false;



vin_sweep_length=length(cadence_data(:,1));
control_voltage=0:0.002 :1.1;
cadence_data_pp=zeros(vin_sweep_length*length(control_voltage),3);
vout_col=2;
row_jmp=0;
for i=1:length(control_voltage)
    cadence_data_pp((row_jmp+1):(row_jmp+vin_sweep_length),1) = control_voltage(i);
    cadence_data_pp((row_jmp+1):(row_jmp+vin_sweep_length),2) = cadence_data(1:vin_sweep_length,1);
    cadence_data_pp((row_jmp+1):(row_jmp+vin_sweep_length),3) = cadence_data(:,vout_col);
    row_jmp = row_jmp + vin_sweep_length;
    vout_col = vout_col + 2;

end



vc = cadence_data_pp(:,1);
vin = cadence_data_pp(:,2);
vout = cadence_data_pp(:,3);
VIN = vin(1:vin_sweep_length,1);
VOUT = reshape(vout,[vin_sweep_length,length(control_voltage)]);

delta = (VIN(2) - VIN(1)) * 0.001;
Truncate = 1000;
sequence_length = 3000;
Number_of_interation = sequence_length + Truncate;



initial_state = 0.5;
input_voltage = initial_state * ones(1,length(control_voltage));
Difference = zeros(Number_of_interation,length(control_voltage));
Diff_sum = zeros(1,length(control_voltage));
output_voltage = zeros(Number_of_interation,length(control_voltage));
output_voltage_del = zeros(Number_of_interation,length(control_voltage));

Vc = repmat(control_voltage,[length(control_voltage),1])';
Vc_unrolled = reshape(Vc,[1,(length(control_voltage))^2]);



for Iteration = 1: Number_of_interation

    input_voltage_del = input_voltage + delta;
    input_voltage_del(input_voltage_del > max(VIN)) = input_voltage_del(input_voltage_del > max(VIN)) - 2 * delta;
    output_voltage(Iteration,:) = interp2(control_voltage,VIN,VOUT,control_voltage,input_voltage);
    output_voltage_del(Iteration,:) = interp2(control_voltage,VIN,VOUT,control_voltage,input_voltage_del);
    Difference(Iteration,:) = log(abs((output_voltage_del(Iteration,:) - output_voltage(Iteration,:)) ./ (input_voltage_del - input_voltage)));
    Difference(Iteration, (Difference(Iteration,:) == -Inf)) = -5;

    if Iteration > Truncate

        Diff_sum = Diff_sum + Difference(Iteration,:);

    end



    input_voltage = output_voltage(Iteration,:);

end



LE = Diff_sum ./ (Number_of_interation - Truncate);
LE_cr = sum(LE > 0) / length(LE);

ALE = mean(LE(LE > 0));
[MLE, ind_MLE] = max(LE);
Vc_MLE = control_voltage(ind_MLE);



fprintf('Chaotic Ratio (LE_cr): %f\n', LE_cr);
fprintf('Average Lyapunov Exponent (ALE): %f\n', ALE);
fprintf('Maximum Lyapunov Exponent (MLE): %f\n', MLE);



metrics = struct('le_cr', LE_cr, 'ale', ALE, 'mle', MLE, 'vc_mle', Vc_MLE);

if do_plots

    figure;

    hold on;

    for i = 1:length(control_voltage)

        current_Vout = cadence_data_pp((i - 1) * vin_sweep_length + 1:i * vin_sweep_length, 3);

        plot(vin(1:vin_sweep_length), current_Vout, 'DisplayName', sprintf('V_c = %.1f V', control_voltage(i)));

    end

    xlabel('Vin (V)', 'FontWeight', 'bold', 'FontSize', 18);
    ylabel('Vout (V)', 'FontWeight', 'bold', 'FontSize', 18);
    title('Transfer Curve for Different Vc Values', 'FontWeight', 'bold', 'FontSize', 20);
    legend show;
    set(gca, 'FontSize', 16);
    xticks(0:0.05:1.1);
    yticks(0:0.1:1.1)
    grid on;
    hold off;

end



if do_plots

    TestMatrix = output_voltage(Truncate + 1:end,:);
    figure;
    plot(control_voltage, output_voltage(Truncate + 1:end,:), '.b', 'markersize', 20);
    x_tick_levels = [0, 0.25, 0.5, 0.75, 1.1];
    y_tick_levels = [0, 0.25, 0.5, 0.75, 1.1];
    xticks(x_tick_levels);
    yticks(y_tick_levels);
    xlim([0, 1.1]);
    ylim([0, 1.1]);
    set(gca, 'FontWeight', 'Bold', 'FontSize', 50, 'box', 'on', 'LineWidth', 5);
    xlabel('Bifurcation parameter, V_{c} (V)', 'FontWeight', 'bold', 'FontSize', 50, 'FontName', 'Times');
    ylabel('Output voltage, V_{o} (V)', 'FontWeight', 'bold', 'FontSize', 50, 'FontName', 'Times');

end



if do_plots
    figure;
    plot(control_voltage, LE, '-r', 'LineWidth', 10);
    x_tick_levels = [0, 0.25, 0.5, 0.75, 1.1];
    y_tick_levels = [-1, -0.5, 0, 0.5, 1];
    xticks(0:0.1:1.1);
    yticks(y_tick_levels);
    xlim([0, 1]);
    ylim([-1, 1]);
    set(gca, 'FontWeight', 'Bold', 'FontSize', 50, 'box', 'on', 'LineWidth', 5);
    xlabel('Bifurcation parameter, V_{c} (V)', 'FontWeight', 'bold', 'FontSize', 50, 'FontName', 'Times');
    ylabel('Lyapunov exponent, \lambda', 'FontWeight', 'bold', 'FontSize', 50, 'FontName', 'Times');

end



if nargin >= 2 && strlength(string(out_json_path)) > 0

    try

        json_text = jsonencode(metrics);
        fid = fopen(out_json_path, 'w');
        fwrite(fid, json_text, 'char');
        fclose(fid);

    catch ME

        warning(ME.message);

    end

end



toc

end
