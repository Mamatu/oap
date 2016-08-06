package ogla.axischart2D;

import java.util.List;
import javax.swing.JPanel;
import ogla.axischart.AxisChartBundle;
import ogla.axischart.lists.BaseList;
import ogla.chart.ChartPanel;
import ogla.core.Help;
import ogla.core.application.Application;
import ogla.core.application.ApplicationBundle;
import ogla.core.event.AnalysisEvent;
import ogla.core.ui.OglaObject;
import ogla.core.util.DefaultChapterImpl;
import ogla.core.util.DefaultDocumentImpl;

/**
 *
 * @author marcin
 */
public class AxisChart2DBundle extends AxisChartBundle {

    private Help.Document doc = new DefaultDocumentImpl("Frame of chart", "chart_frame_help.html", AxisChart2DBundle.this.getClass());
    private Help.Document doc1 = new DefaultDocumentImpl("Frame of data", "data_frame_help.html", AxisChart2DBundle.this.getClass());
    private Help.Document doc2 = new DefaultDocumentImpl("Frame of properties", "properties_frame_help.html", AxisChart2DBundle.this.getClass());
    private Help.Document doc3 = new DefaultDocumentImpl("History", "history.html", AxisChart2DBundle.this.getClass());
    private Help.Document doc4 = new DefaultDocumentImpl("Groups", "group.html", AxisChart2DBundle.this.getClass());
    private Help.Document doc5 = new DefaultDocumentImpl("Repository", "repository.html", AxisChart2DBundle.this.getClass());
    private DefaultChapterImpl root = new DefaultChapterImpl("Axis chart");
    private DefaultChapterImpl framesChapter = new DefaultChapterImpl("About frames");
    private Help help = new Help() {

        public Chapter getRootChapter() {
            return root;
        }
    };

    public Help getHelp() {
        return help;
    }

    public boolean canBeSaved() {
        return true;
    }

    public Application load(byte[] bytes) {
        ApplicationImpl applicationImpl = new ApplicationImpl();
        AxisChart2D axisChart = new AxisChart2D(applicationImpl.chartPanel,
                baseLists);
        applicationImpl.axisChart = axisChart;
        axisChart.load(bytes);
        return applicationImpl;
    }

    public String getSymbolicName() {
        return "axis_chart_2D_((^^&*#(";
    }

    public AnalysisEvent[] getEvents() {
        return null;
    }

    private class ApplicationImpl implements Application {

        class OglaObjectImpl extends OglaObject {

            public OglaObjectImpl() {
                super("", null, null);
            }
        }

        ChartPanel chartPanel = new ChartPanel();
        AxisChart2D axisChart = null;
        OglaObjectImpl oglaObjectImpl = new OglaObjectImpl();

        public JPanel getAppPanel() {
            return chartPanel;
        }

        public byte[] save() {
            return axisChart.save();
        }

        public ApplicationBundle getBundle() {
            return AxisChart2DBundle.this;
        }

        public OglaObject getOglaObject(int index) {
            if (index != 0) {
                return null;
            }
            return oglaObjectImpl;
        }

        public int getOglaObjectsCount() {
            return 1;
        }
    }
    protected List<AxisChart2D> axisChartes;
    private List<BaseList> baseLists;

    public AxisChart2DBundle(List<BaseList> baseLists) {
        root.documents.add(doc5);
        root.documents.add(doc3);
        root.documents.add(doc4);
        framesChapter.documents.add(doc);
        framesChapter.documents.add(doc1);
        framesChapter.documents.add(doc2);
        root.subchapters.add(framesChapter);
        this.baseLists = baseLists;
    }

    public Application newApplication() {
        ApplicationImpl applicationImpl = new ApplicationImpl();
        AxisChart2D axisChart = new AxisChart2D(applicationImpl.chartPanel,
                baseLists);
        applicationImpl.axisChart = axisChart;
        return applicationImpl;
    }

    public String getLabel() {
        return "Axis Chart 2D";
    }

    public String getDescription() {
        return "";
    }
}
