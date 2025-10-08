package org.example;

import org.jlab.jnp.hipo4.data.Bank;
import org.jlab.jnp.hipo4.data.Event;
import org.jlab.jnp.hipo4.io.HipoReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

//mvn exec:java -Dexec.mainClass="org.example.DenoiseExtractor"

public class DenoiseExtractor {

    private static final int NLAYERS = 36;
    private static final int NWIRES  = 112;

    private int[][] dcHits = new int[NLAYERS][NWIRES];
    private int[][] tbHits = new int[NLAYERS][NWIRES];

    private void resetArrays() {
        for (int l = 0; l < NLAYERS; l++) {
            for (int w = 0; w < NWIRES; w++) {
                dcHits[l][w] = 0;
                tbHits[l][w] = 0;
            }
        }
    }

    public void processDirectory(String inputDir, String outPrefix, int maxEvents) throws IOException {
        File dir = new File(inputDir);
        File[] files = dir.listFiles((d, name) -> name.endsWith(".hipo"));

        if (files == null) {
            System.out.println("No .hipo files found in directory " + inputDir);
            return;
        }

        int evCountTotal = 0;

        // Loop sector by sector to reduce memory overhead
        for (int sector = 1; sector <= 6; sector++) {
            String outName = String.format("%s_sector%d.csv", outPrefix, sector);
            try (FileWriter writer = new FileWriter(outName)) {

                int evCountSector = 0, rep=0;

                // Loop over all files for this sector
                for (File f : files) {
                    if (evCountSector >= maxEvents) break;

                    System.out.printf("Sector %d: processing file %s%n", sector, f.getName());
                    HipoReader reader = new HipoReader();
                    reader.open(f.getAbsolutePath());
                    Event event = new Event();

                    Bank dcTDC = reader.getBank("DC::tdc");
                    Bank tbHit = reader.getBank("TimeBasedTrkg::TBHits");

                    while (reader.hasNext() && evCountSector < maxEvents) {
                        reader.nextEvent(event);
                        event.read(dcTDC);
                        event.read(tbHit);

                        resetArrays();

                        // Fill DC::tdc for this sector
                        for (int row = 0; row < dcTDC.getRows(); row++) {
                            int sec   = dcTDC.getInt("sector", row);
                            int layer = dcTDC.getInt("layer", row);
                            int comp  = dcTDC.getInt("component", row);

                            if (sec == sector && valid(layer, comp)) {
                                dcHits[layer - 1][comp - 1] = 1;
                            }
                        }

                        // Fill TBHits for this sector
                        for (int row = 0; row < tbHit.getRows(); row++) {
                            int sec   = tbHit.getInt("sector", row);
                            int layer = tbHit.getInt("layer", row);
                            int slayer = tbHit.getInt("superlayer", row);
                            int wire  = tbHit.getInt("wire", row);

                            if (sec == sector && valid(layer, wire)) {
                                int index = (slayer - 1) * 6 + layer;
                                tbHits[index-1][wire - 1] = 1;
                            }
                        }

                        // Write event if TBHits has hits
                        if (hasHits(tbHits)) {
                            writeEventBlock(writer, dcHits, tbHits);
                            evCountSector++;
                        }

                        evCountTotal++;
                        if ((evCountSector % 10000) == 0 && evCountSector!=rep) {
                            System.out.printf("\n***** Sector %d: processed %d events so far... *****\n\n",
                                              sector, evCountSector);
                            rep=evCountSector;
                        }
                    }
                    reader.close();
                }

                System.out.printf("Finished sector %d, wrote %d events%n", sector, evCountSector);
            }
        }

        System.out.printf("Processed %d events in total%n", evCountTotal);
    }

    private boolean valid(int layer, int wire) {
        return (layer >= 1 && layer <= NLAYERS && wire >= 1 && wire <= NWIRES);
    }

    private boolean hasHits(int[][] arr) {
        for (int l = 0; l < NLAYERS; l++) {
            for (int w = 0; w < NWIRES; w++) {
                if (arr[l][w] != 0) return true;
            }
        }
        return false;
    }

    private void writeEventBlock(FileWriter fw, int[][] arr1, int[][] arr2) throws IOException {
        // DC::tdc
        for (int l = 0; l < NLAYERS; l++) {
            for (int w = 0; w < NWIRES; w++) {
                fw.write(arr1[l][w] + (w < NWIRES - 1 ? "," : ""));
            }
            fw.write("\n");
        }
        fw.write("\n");

        // TBHits
        for (int l = 0; l < NLAYERS; l++) {
            for (int w = 0; w < NWIRES; w++) {
                fw.write(arr2[l][w] + (w < NWIRES - 1 ? "," : ""));
            }
            fw.write("\n");
        }
        fw.write("\n\n");
    }

    public static void main(String[] args) throws IOException {
        String inputDir = "/volatile/clas12/users/caot/experiment/aiAssistedPlusTracking_iss471/rga_fall2018/full/recon/005197/";
        String prefix = "run5197";

        DenoiseExtractor de = new DenoiseExtractor();
        de.processDirectory(inputDir, prefix, 600000);
    }
}
