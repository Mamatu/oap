package ogla.axischart;

import ogla.axischart.plugin.plotmanager.PlotManager;
import ogla.core.BasicInformation;
import ogla.core.data.DataRepository;

public interface Displayer extends BasicInformation {

    public int getNumberOfDataRepositories();

    public DataRepository getDataRepository();

    public DataRepository getDataRepository(int index);

    public String getLabel();

    public PlotManager getPlotManager();

    public boolean isVisible();
}
