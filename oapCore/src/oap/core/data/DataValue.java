package ogla.core.data;

import java.io.Serializable;

/**
 * Class whose instance represent single variable.
 */
public interface DataValue extends Serializable {

    public static final DataValue Empty = new DataValue() {

        public String getLabel() {
            return "";
        }

        public Number getNumber() {
            return null;
        }

        @Override
        public String toString() {
            return "Empty(" + super.toString() + ")";
        }
    };
    public static final DataValue Zero = new DataValue() {

        public String getLabel() {
            return "0";
        }

        public Number getNumber() {
            return 0;
        }
    };

    /**
     * Label of value which should displayed on chart.
     * @return
     */
    public String getLabel();

    /**
     * Get value of data.
     * @return
     */
    public Number getNumber();
}
