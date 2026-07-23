import UploadDetect from "../components/UploadDetect.jsx";
import { detectApi } from "../services/api.js";

export default function ImageDetection() {
  return (
    <div>
      <h1 className="mb-6 text-2xl font-bold">Image Detection</h1>
      <UploadDetect
        title="Upload Image"
        accept="image/*"
        onDetect={async (fd) => (await detectApi.image(fd)).data}
      />
    </div>
  );
}
