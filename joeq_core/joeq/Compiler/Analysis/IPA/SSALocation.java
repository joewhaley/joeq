package Compil3r.Analysis.IPA;

/**
 * @author Vladimir Livshits
 */
public interface SSALocation {

    /**
     * 	We need to have "abstract" temporary locations for IPSSA construction purposes 
     * that do not necessarily correspond to anything tangible. 
     * */
    public static class Temporary implements SSALocation {
        private Temporary() {
            // there's no underlying node
        }

        // There's only one Temporary location -- use the FACTORY to retrieve it	
        public static class FACTORY {
            private static Temporary _sample = new Temporary();
            public static Temporary get() {
                return _sample;
            }
        }
    }
}
