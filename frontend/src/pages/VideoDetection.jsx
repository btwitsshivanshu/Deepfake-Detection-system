import UploadDetect from "../components/UploadDetect.jsx";
import { detectApi } from "../services/api.js";

export default function VideoDetection() {
  return (
    <div>
      <h1 className="mb-6 text-2xl font-bold">Video Detection</h1>
      <UploadDetect
        title="Upload Video"
        accept="video/*"
        onDetect={async (fd) => (await detectApi.video(fd)).data}
      />
    </div>
  );
}
