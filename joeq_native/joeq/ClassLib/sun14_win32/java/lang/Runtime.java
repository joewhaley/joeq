package ClassLib.sun14_win32.java.lang;

public class Runtime {

    public static native Runtime getRuntime();
    synchronized native void loadLibrary0(Class fromClass, String libname);
}
