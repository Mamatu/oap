package ogla.core.util;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Class to reading some values from string of bytes.
 * @author marcin
 */
public class Reader {

    /**
     * Exception which is thrown if you want to index which is higher than string bytes length.
     */
    public class EndOfBufferException extends Exception {

        public EndOfBufferException(int pos, int size, int length) {
            super("End of buffer. Size: "
                    + String.valueOf(size) + ". Index: " + String.valueOf(pos)
                    + ". Length: " + String.valueOf(length) + "\n");
        }
    }
    private int pos = 0;
    private byte[] bytes;
    private byte[] word = new byte[4];
    private byte[] doubleWord = new byte[8];
    private byte[] word1 = new byte[4];
    private byte[] doubleWord1 = new byte[8];

    public Reader() {
    }

    public Reader(byte[] bytes) {
        this.bytes = bytes;
    }

    /**
     * Set array of bytes which is used by reading methods.
     * @param bytes
     */
    public Reader setBytes(byte[] bytes) {
        this.bytes = bytes;
        pos = 0;
        return this;
    }

    private void check(int length) throws EndOfBufferException {
        if (bytes.length < pos + length) {
            throw new EndOfBufferException(pos, bytes.length, length);
        }
    }

    /**
     * Read string of bytes (to array) which length is equal to length of array.
     * @param array
     * @throws ogla.excore.util.Reader.EndOfBufferException
     */
    public void readBytes(byte[] array) throws EndOfBufferException {
        check(array.length);
        System.arraycopy(bytes, pos, array, 0, array.length);
        pos += array.length;
    }

    /**
     * Read boolean value.
     * @return
     * @throws ogla.excore.util.Reader.EndOfBufferException
     */
    public boolean readBoolean() throws EndOfBufferException {
        check(1);
        boolean b = false;
        if (bytes[pos] == 1) {
            b = true;
        }
        pos++;
        return b;
    }

    /**
     * Read string of chars.
     * @param length
     * @return string of chars
     * @throws ogla.excore.util.Reader.EndOfBufferException
     */
    public String readString(int length) throws EndOfBufferException {
        check(length);
        StringBuilder builder = new StringBuilder();
        for (int fa = 0; fa < length; fa++) {
            builder.append((char) bytes[pos]);
            pos++;
        }
        return builder.toString();
    }

    /**
     * Read float but first byte is treated as last and so on.
     * @return read variable
     * @throws ogla.excore.util.Reader.EndOfBufferException
     * @throws IOException
     */
    public float readFloatReverse() throws EndOfBufferException, IOException {
        check(4);
        System.arraycopy(bytes, pos, word1, 0, 4);
        for (int fa = 0; fa < 4; fa++) {
            word[fa] = word1[3 - fa];
        }
        pos += 4;
        return ByteBuffer.wrap(word).getFloat();
    }

    /**
     * Read float
     * @return read variable
     * @throws ogla.excore.util.Reader.EndOfBufferException
     * @throws IOException
     */
    public float readFloat() throws EndOfBufferException, IOException {
        check(4);
        System.arraycopy(bytes, pos, word1, 0, 4);
        for (int fa = 0; fa < 4; fa++) {
            word[fa] = word1[fa];
        }
        pos += 4;
        return ByteBuffer.wrap(word).getFloat();
    }

    public double readDouble() throws EndOfBufferException, IOException {
        check(8);
        System.arraycopy(bytes, pos, doubleWord1, 0, 4);
        for (int fa = 0; fa < 8; fa++) {
            doubleWord[fa] = doubleWord1[fa];
        }
        pos += 8;
        return ByteBuffer.wrap(doubleWord).getDouble();
    }

    public double readDoubleReverse() throws EndOfBufferException, IOException {
        check(8);
        System.arraycopy(bytes, pos, doubleWord1, 0, 4);
        for (int fa = 0; fa < 8; fa++) {
            doubleWord[8 - fa] = doubleWord1[fa];
        }
        pos += 8;
        return ByteBuffer.wrap(doubleWord).getDouble();
    }

    /**
     * Read integer but first byte is treated as last and so on.
     * @return read variable
     * @throws ogla.excore.util.Reader.EndOfBufferException
     * @throws IOException
     */
    public int readIntReverse() throws EndOfBufferException, IOException {
        check(4);
        System.arraycopy(bytes, pos, word1, 0, 4);
        for (int fa = 0; fa < 4; fa++) {
            word[fa] = word1[3 - fa];
        }
        pos += 4;
        return ByteBuffer.wrap(word).getInt();
    }

    /**
     * Read integer
     * @return read variable
     * @throws ogla.excore.util.Reader.EndOfBufferException
     * @throws IOException
     */
    public int readInt() throws EndOfBufferException, IOException {
        check(4);
        System.arraycopy(bytes, pos, word1, 0, 4);
        for (int fa = 0; fa < 4; fa++) {
            word[fa] = word1[fa];
        }
        pos += 4;
        return ByteBuffer.wrap(word).getInt();
    }

    public int getPosition() {
        return pos;
    }

    public int getSize() {
        return this.bytes.length;
    }

    public void skip(int offset) {
        pos += offset;
    }

    public int getRemainingCapacity() {
        if (bytes.length == 0) {
            return 0;
        }
        return bytes.length - pos - 1;
    }
}
