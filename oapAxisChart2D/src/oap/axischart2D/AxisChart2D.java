package ogla.axischart2D;

import ogla.axischart2D.util.Recorder;
import ogla.axischart2D.coordinates.CoordinateTransformer;
import ogla.axischart2D.coordinates.CoordinateXAxis;
import ogla.axischart2D.coordinates.CoordinateYAxis;
import ogla.axischart2D.gui.ComponentsPanel;
import ogla.axischart2D.gui.DataPanel;
import ogla.axischart2D.gui.PropertiesPanel;
import ogla.axischart2D.popup.FramesPopupMenu;
import javax.swing.JPanel;
import org.osgi.framework.BundleContext;
import ogla.axischart.AxisChart;
import ogla.axischart.Displayer;
import ogla.axischart.plugin.image.ImageComponent;
import ogla.axischart.plugin.image.ImageComponentBundle;
import ogla.axischart.lists.BaseList;
import ogla.axischart.lists.ListDisplayer;
import ogla.axischart.lists.ListImageComponentBundle;
import ogla.axischart.lists.ListInstalledImageComponent;
import ogla.axischart.lists.ListInterface;
import ogla.axischart.lists.ListListener;
import ogla.axischart.util.DefaultDisplayingStackImpl;
import ogla.axischart2D.gui.DisplayingStackPanel;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponentBundle;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponent;
import ogla.axischart2D.plugin.plotmanager.AxisChart2DPlotManagerBundle;
import ogla.axischart2D.lists.AxisChart2DComponentBundleList;
import ogla.axischart2D.lists.ListAxisChart2DInstalledComponent;
import ogla.axischart2D.lists.PlotManagerBundleList;
import ogla.axischart2D.util.AxisIndices;
import ogla.axischart2D.util.DisplayerExtendedImpl;
import ogla.chart.IOEntity;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import ogla.axischart2D.gui.ChartCanvas;
import ogla.chart.ChartPanel;
import ogla.chart.gui.InternalFrame;
import ogla.core.data.DataBlock;
import ogla.core.data.DataBundle;
import ogla.core.data.DataRepository;
import ogla.core.data.listeners.RepositoryListener;

public class AxisChart2D extends AxisChart {

    void addIOEntityProxy(IOEntity iOEntity) {
        super.addIOEntity(iOEntity);
    }
    protected int firstIndex = 0;
    protected CoordinateTransformer[] transformers = {new CoordinateXAxis(), new CoordinateYAxis()};
    protected boolean automaticalRefresh = false;
    protected boolean automaticalAdjusting = false;
    protected boolean automaticalShiftingToEnd = false;
    protected boolean useDynamicStep = false;
    protected Recorder recorder = new Recorder(this);

    public void useDynamicStep(boolean b) {
        useDynamicStep = b;
        if (!b) {
            propertiesPanel.createTicksFromUser();
        }
        this.repaintChartSurface();
    }

    public Recorder getRecorder() {
        return recorder;
    }

    public String getLabel() {
        return "Axis Chart 2D";
    }

    public String getDescription() {
        return "";
    }

    private class RepositoryListenerImpl implements RepositoryListener {

        public void listenerIsAdded(DataBlock repository) {
        }

        public void listenerIsRemoved(DataBlock repository) {
        }

        private Displayer getDisplayer(DataBlock dataBlock) {
            for (int fa = 0; fa < listDisplayer.size(); fa++) {
                Displayer displayer = listDisplayer.get(fa);
                for (int fb = 0; fb < displayer.getNumberOfDataRepositories(); fb++) {
                    DataRepository dataRepository = displayer.getDataRepository(fb);
                    if (dataRepository == dataBlock.getDataRepository()) {
                        return displayer;
                    }
                }
            }
            return null;
        }

        private float[] findMinMaxY(DataBlock dataBlock, float[] a) {
            int cy = axisIndices.getYIndex(dataBlock);
            float min = dataBlock.get(0, cy).getNumber().floatValue();
            float max = dataBlock.get(0, cy).getNumber().floatValue();
            int size = dataBlock.rows();
            for (int fa = 1; fa < size; fa++) {
                float y = dataBlock.get(fa, cy).getNumber().floatValue();
                if (y < min) {
                    min = y;
                }
                if (y > max) {
                    max = y;
                }
            }
            a[0] = min;
            a[1] = max;
            return a;
        }
        private float[] temp = new float[2];
        private Displayer currentDisplayer = null;
        private AxisIndices axisIndices = null;

        public void isChangedSizeAndData(DataBlock dataBlock) {
            currentDisplayer = getDisplayer(dataBlock);
            DisplayerExtendedImpl dei = (DisplayerExtendedImpl) currentDisplayer;
            axisIndices = dei.getAxisIndices();
            int cx = axisIndices.getXIndex(dataBlock);
            if (currentDisplayer == null) {
                return;
            }
            if (automaticalAdjusting) {
                int size = dataBlock.rows();

                float maxX = dataBlock.get(size - 1, cx).getNumber().floatValue();
                float minX = dataBlock.get(0, cx).getNumber().floatValue();
                float[] m = findMinMaxY(dataBlock, temp);
                float minY = m[0];
                float maxY = m[1];
                xTicks.setGreatestTick(maxX);
                xTicks.setLowestTick(minX);
                yTicks.setGreatestTick(maxY);
                yTicks.setLowestTick(minY);
                if (useDynamicStep) {
                    xTicks.setDisplayedTicks(minX, maxX, xTicks.getDynamicStep());
                    yTicks.setDisplayedTicks(minY, maxY, yTicks.getDynamicStep());
                }

            } else if (automaticalShiftingToEnd) {
                float min = xTicks.getLowestTick().floatValue();
                float max = xTicks.getGreatestTick().floatValue();
                float length = max - min;
                float x = dataBlock.get(dataBlock.rows() - 1, cx).getNumber().floatValue();
                float x1 = x - length;
                float[] m = findMinMaxY(dataBlock, temp);
                if (x1 >= min) {
                    xTicks.setLowestTick(x1);
                    xTicks.setGreatestTick(x);
                    yTicks.setLowestTick(m[0]);
                    yTicks.setGreatestTick(m[1]);
                    if (useDynamicStep) {
                        xTicks.setDisplayedTicks(x1, x, xTicks.getDynamicStep());
                        yTicks.setDisplayedTicks(m[0], m[1], yTicks.getDynamicStep());
                    }
                }
            }
            repaintChartSurface();
        }

        public void isChangedData(DataBlock repository) {
            repaintChartSurface();
        }
    }

    void registerLoadedBundleData(DataBundle bundleData) {
        bundleDataPanel.fillList(bundleData);
    }
    protected RepositoryListener repositoryListener = null;
    private RepositoryListenerImpl repositoryListenerImpl = new RepositoryListenerImpl();

    public RepositoryListener getRepositoryListener() {
        return repositoryListener;
    }

    public void setAutomaticalRefresh(boolean b) {
        automaticalRefresh = b;
        ListDisplayer listDisplayerInfo = listsContainer.get(ListDisplayer.class);
        if (b) {
            repositoryListener = repositoryListenerImpl;
            for (int fa = 0; fa < listDisplayerInfo.size(); fa++) {
                listDisplayerInfo.get(fa).getDataRepository().addRepositoryListener(repositoryListener);
            }
        } else {
            for (int fa = 0; fa < listDisplayerInfo.size(); fa++) {
                listDisplayerInfo.get(fa).getDataRepository().removeRepositoryListener(repositoryListener);
            }
            repositoryListener = null;
        }
    }

    public void setAutomaticalAdjusting(boolean b) {
        automaticalAdjusting = b;
    }

    public void setAutomaticalShifting(boolean b) {
        automaticalShiftingToEnd = b;
    }

    /**
     * Get transformer for x axis.
     * @return transformer for x axis
     */
    public CoordinateTransformer getTransformerOfXAxis() {
        return transformers[0];
    }

    /**
     * Get transformer for y axis.
     * @return transformer for y axis
     */
    public CoordinateTransformer getTransformerOfYAxis() {
        return transformers[1];
    }

    @Override
    public void repaintChartSurface() {
        chartPanel.repaint();
        super.repaintChartSurface();
        recorder.rec();
    }
    protected ChartPanel chartPanel = null;
    protected PropertiesPanel propertiesPanel = null;
    protected DisplayingStackPanel displayStackPanel = null;
    protected DataPanel bundleDataPanel = null;
    protected ComponentsPanel componentsPanel = null;
    protected ChartCanvas chartCanvas = null;
    protected FramesPopupMenu framesPopupMenu;
    protected BundleContext bundleContext;
    protected InternalFrame[] frames;

    private boolean put(ImageComponentBundle bundleImageComponent, ImageComponent imageComponent) {
        if (imageBundleImageComponents.containsKey(bundleImageComponent)) {
            imageBundleImageComponents.get(bundleImageComponent).add(imageComponent);
            return false;
        } else {
            List<ImageComponent> list = new ArrayList<ImageComponent>();
            list.add(imageComponent);
            imageBundleImageComponents.put(bundleImageComponent, list);
            return true;
        }
    }
    private Map<ImageComponentBundle, List<ImageComponent>> imageBundleImageComponents = new HashMap<ImageComponentBundle, List<ImageComponent>>();
    private IOEntities iOEntities = null;
    private ListAxisChart2DInstalledComponent listInstalledChartComponent = new ListAxisChart2DInstalledComponent(this);

    public AxisChart2D(ChartPanel chartPanel,
            List<BaseList> baseLists) {
        super(chartPanel.getChartSurface(), baseLists);
        this.listsContainer.add(listInstalledChartComponent);
        this.chartPanel = chartPanel;
        this.displayStack = new DefaultDisplayingStackImpl();
        frames = initializeGUI();

        initializeChart();

        framesPopupMenu = new FramesPopupMenu(chartPanel.getDesktop(), frames[0], frames[2], frames[3], frames[1]);
        chartPanel.setPopup(framesPopupMenu);
        iOEntities = new IOEntities(this, listsContainer);

        ListImageComponentBundle listImageComponentBundle = listsContainer.get(ListImageComponentBundle.class);
        listImageComponentBundle.addListListener(new ListListener<ImageComponentBundle>() {

            public void isAdded(ImageComponentBundle t) {

                ImageComponent imageComponent = t.newImageComponent();
                if (put(t, t.newImageComponent())) {
                    ListInstalledImageComponent list = listsContainer.get(ListInstalledImageComponent.class);
                    list.add(imageComponent);
                }

            }

            public void isRemoved(ImageComponentBundle t) {
                for (ImageComponent imageComponent : imageBundleImageComponents.get(t)) {
                    ListInstalledImageComponent list = listsContainer.get(ListInstalledImageComponent.class);
                    list.remove(imageComponent);
                }
            }
        });

        ListDisplayer listDisplayerInfo = listsContainer.get(ListDisplayer.class);
        listDisplayerInfo.addListListener(new ListListener<Displayer>() {

            public void isAdded(Displayer t) {
                DisplayerImpl displayer = (DisplayerImpl) t;
                AxisChart2D.this.displayStack.add(displayer);
            }

            public void isRemoved(Displayer t) {
                DisplayerImpl displayer = (DisplayerImpl) t;
                AxisChart2D.this.displayStack.remove(displayer);
            }
        });
        //listsContainer.getListInstalledImageComponent().add(imageComponent, imageComponent.getBundle().getLabel());
    }

    private InternalFrame[] initializeGUI() {
        InternalFrame[] frames = new InternalFrame[5];
        xTicks = new Ticks();
        yTicks = new Ticks();
        chartCanvas = new ChartCanvas(this, listsContainer);
        frames[0] = newInternalFrame(chartCanvas, "Chart");
        propertiesPanel = new PropertiesPanel(this);
        frames[1] = newInternalFrame(propertiesPanel, "Properties");
        bundleDataPanel = new DataPanel(this, listsContainer);
        frames[2] = newInternalFrame(bundleDataPanel, "Data");
        componentsPanel = new ComponentsPanel(this, listsContainer, displayStack);
        frames[3] = newInternalFrame(componentsPanel, "Chart's components");
        displayStackPanel = new DisplayingStackPanel(this, displayStack);
        frames[4] = newInternalFrame(displayStackPanel, "Display stack");
        chartPanel.getDesktop().revalidate();
        chartPanel.getDesktop().repaint();
        return frames;
    }

    private InternalFrame newInternalFrame(JPanel panel, String title) {
        return new InternalFrame(title, chartPanel, panel);
    }

    private void initializeChart() {
        this.repaintChartSurface();
    }
    protected Ticks xTicks;
    protected Ticks yTicks;

    public Ticks getXTicks() {
        return xTicks;
    }

    public Ticks getYTicks() {
        return yTicks;
    }
    protected DefaultDisplayingStackImpl displayStack;

    /**
     * Get display stack which store info about displayed components.
     * @return
     */
    public DisplayingStack getDisplayStack() {
        return displayStack;
    }

    /**
     * Get list of displayers.
     * @return
     */
    public ListInterface<Displayer> getListDisplayerInfo() {
        return listsContainer.get(ListDisplayer.class);
    }

    /**
     * Get list of ChartComponent's bundles.
     * @return
     */
    public ListInterface<AxisChart2DComponentBundle> getListChartComponentBundle() {
        return listsContainer.get(AxisChart2DComponentBundleList.class);
    }

    /**
     * Get list of installed ChartComponents.
     * @return
     */
    public ListInterface<AxisChart2DComponent> getListInstalledChartComponent() {
        return listInstalledChartComponent;
    }

    /**
     * Get list of PlotManagers' bundles.
     * @return
     */
    public ListInterface<AxisChart2DPlotManagerBundle> getListPlotManagerBundle() {
        return listsContainer.get(PlotManagerBundleList.class);
    }
}


