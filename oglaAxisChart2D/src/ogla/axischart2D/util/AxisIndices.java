package ogla.axischart2D.util;

import ogla.axischart.AxisChart;
import ogla.core.data.DataBlock;

public class AxisIndices {

    public AxisIndices() {
    }

    public AxisIndices(int indexx, int typex, int indexy, int typey) {
        this.indexX = indexx;
        this.sideX = typex;
        this.indexY = indexy;
        this.sideY = typey;
    }

    public AxisIndices(AxisIndices axisIndices) {
        axisIndices.giveTo(this);
    }
    public static final int FromBegin = 0;
    public static final int FromEnd = 1;
    private int sideX = AxisIndices.FromBegin;
    private int sideY = AxisIndices.FromEnd;
    private int indexX = 1;
    private int indexY = 1;
    public boolean xIsChanged = false;
    public boolean yIsChanged = false;

    public void giveTo(AxisIndices axisIndices) {
        axisIndices.setIndexX(this.indexX);
        axisIndices.setIndexY(this.indexY);
        axisIndices.setSideX(this.sideX);
        axisIndices.setSideY(this.sideY);
    }

    public void setIndexX(int index) {
        this.indexX = index;
        xIsChanged = true;
    }

    public void setIndexY(int index) {
        this.indexY = index;
        yIsChanged = true;
    }

    public void setSideX(int type) {
        this.sideX = type;
        xIsChanged = true;
    }

    public void setSideY(int type) {
        this.sideY = type;
        yIsChanged = true;
    }

    public int getSideX() {
        return sideX;
    }

    public int getSideY() {
        return sideY;
    }

    public class OutOfRangeException extends Exception {

        public OutOfRangeException(int index, int size) {
            super("Index: " + String.valueOf(index) + " is not in the interval [0," + String.valueOf(size) + ")");
        }
    }

    public int getRawXIndex() {
        return indexX;
    }

    public int getRawYIndex() {
        return indexY;
    }

    public int getXIndex(DataBlock dataBlock) {
        if (sideX == AxisIndices.FromBegin) {
            int t = indexX - 1;
            if (0 <= t && t < dataBlock.columns()) {
                return t;
            } else {
                return 0;
            }
        } else {
            int t = dataBlock.columns() - indexX;
            if (0 <= t && t < dataBlock.columns()) {
                return t;
            } else {
                return dataBlock.columns() - 1;
            }
        }
    }

    public int getYIndex(DataBlock dataBlock) {
        if (sideY == AxisIndices.FromBegin) {
            int t = indexY - 1;
            if (0 <= t && t < dataBlock.columns()) {
                return t;
            } else {
                return 0;
            }
        } else {
            int t = dataBlock.columns() - indexY;
            if (0 <= t && t < dataBlock.columns()) {
                return t;
            } else {
                return dataBlock.columns() - 1;
            }
        }
    }
}
