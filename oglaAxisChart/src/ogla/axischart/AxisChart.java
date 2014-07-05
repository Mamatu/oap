package ogla.axischart;

import java.util.List;
import ogla.axischart.lists.BaseList;
import ogla.axischart.lists.ListDataBundle;
import ogla.axischart.lists.ListDataRepository;
import ogla.axischart.lists.ListDisplayer;
import ogla.axischart.lists.ListImageComponentBundle;
import ogla.axischart.lists.ListInstalledImageComponent;
import ogla.axischart.lists.ListInterface;
import ogla.axischart.lists.ListsContainer;
import ogla.axischart.plugin.image.ImageComponent;
import ogla.axischart.plugin.image.ImageComponentBundle;
import ogla.chart.Chart;
import ogla.chart.ChartSurface;
import ogla.chart.IOEntity;
import ogla.core.data.DataBundle;
import ogla.core.data.DataRepository;


public abstract class AxisChart extends Chart {

    protected void addIOEntity(IOEntity iOEntity) {
        this.iOManager.iOEntities.add(iOEntity);
    }
    protected ListsContainer listsContainer = new ListsContainer();
    protected ListInstalledImageComponent listInstalledImageComponent = new ListInstalledImageComponent();
    protected ListDataRepository listDataRepositoriesGroup = new ListDataRepository();
    protected ListDisplayer listDisplayer = new ListDisplayer();

    public AxisChart(ChartSurface chartSurface, List<BaseList> baseLists) {
        super(chartSurface);
        for (BaseList baseList : baseLists) {
            listsContainer.add(baseList);
        }
        listsContainer.add(listInstalledImageComponent);
        listsContainer.add(listDataRepositoriesGroup);
        listsContainer.add(listDisplayer);
        new IOEntities(this, listsContainer);
    }

    public abstract ListInterface<Displayer> getListDisplayerInfo();

    /**
     * Get list of DataRepositoriesGroup object
     * @return list of DataRepository object
     */
    public ListInterface<DataRepository> getListDataRepository() {
        return listDataRepositoriesGroup;
    }

    /**
     * Get list of InstalledImageComponent object
     * @return list of InstalledImageComponent object
     */
    public ListInterface<ImageComponent> getListInstalledImageComponent() {
        return listInstalledImageComponent;
    }

    /**
     * Get list of DataBundle object
     * @return list of DataBundle object
     */
    public ListInterface<DataBundle> getListDataBundle() {
        return listsContainer.get(ListDataBundle.class);
    }

    /**
     * Get list of ImageComponentBundle object
     * @return list of ImageComponentBundle object
     */
    public ListInterface<ImageComponentBundle> getListImageComponentBundle() {
        return listsContainer.get(ListImageComponentBundle.class);
    }

    public ListInterface<Displayer> getListDislpayer() {
        return listDisplayer;
    }
}
