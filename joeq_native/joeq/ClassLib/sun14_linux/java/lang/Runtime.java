package ClassLib.sun14_linux.java.lang;

public class Runtime {

    public native static Runtime getRuntime();
    synchronized native void loadLibrary0(Class fromClass, String libname);
}
