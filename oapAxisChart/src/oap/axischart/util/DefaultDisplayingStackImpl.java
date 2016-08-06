package ogla.axischart.util;

import ogla.chart.Chart.DisplayingStack;
import ogla.chart.DrawableOnChart;
import java.util.ArrayList;
import java.util.List;
import javax.swing.DefaultListModel;
import javax.swing.ListModel;

/**
 *
 * @author marcin
 */
public class DefaultDisplayingStackImpl extends DisplayingStack {

    public DefaultDisplayingStackImpl() {
    }
    private DefaultListModel defaultListModel = new DefaultListModel();

    public ListModel getListModel() {
        return defaultListModel;
    }

    private class Element {

        public Element(DrawableOnChart drawableOnChart) {
            this.drawableOnChart = drawableOnChart;
        }

        @Override
        public String toString() {
            return drawableOnChart.getLabel();
        }
        public DrawableOnChart drawableOnChart;
    }

    private int getIndex(DrawableOnChart obj) {
        for (int fa = 0; fa < defaultListModel.getSize(); fa++) {
            Element element = (Element) defaultListModel.get(fa);
            if (element.drawableOnChart == obj) {
                return fa;
            }
        }
        return -1;
    }
    public void add(DrawableOnChart obj) {
        defaultListModel.add(0, new Element(obj));
    }

    public void add(DrawableOnChart obj, int index) {
        defaultListModel.add(index, new Element(obj));
    }

    public void remove(DrawableOnChart obj) {
        int index = getIndex(obj);
        if (index != -1) {
            defaultListModel.remove(index);
        }
    }

    public void down(int index) {
        if (index >= defaultListModel.size() - 1) {
            return;
        }
        Element e = (Element) defaultListModel.get(index);
        Element e1 = (Element) defaultListModel.get(index + 1);
        defaultListModel.setElementAt(e, index + 1);
        defaultListModel.setElementAt(e1, index);
    }

    public void up(int index) {
        if (index <= 0) {
            return;
        }
        Element e1 = (Element) defaultListModel.get(index);
        Element e = (Element) defaultListModel.get(index - 1);
        defaultListModel.setElementAt(e1, index - 1);
        defaultListModel.setElementAt(e, index);
    }

    @Override
    protected int size() {
        return defaultListModel.size();
    }

    @Override
    protected DrawableOnChart get(int index) {
        Element element = (Element) defaultListModel.get(defaultListModel.size() - index - 1);
        return element.drawableOnChart;
    }
}
